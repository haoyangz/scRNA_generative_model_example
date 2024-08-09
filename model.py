import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from scvi.distributions import ZeroInflatedNegativeBinomial
from torch import nn
from torch.distributions import Categorical, Normal, kl_divergence
from utils import broadcast_labels, reparameterize


class MYVAE(L.LightningModule):
    """
    This model implements a VAE with the following generative process and variational approximation
    p(x|z1,z2,y,l) = p(x|z1,l)p(z1|z2,y)p(y)p(z2)p(l)
    q(z1,z2,y,l|x) = q(z1|x)q(y|z1)q(l|x)q(z2|z1,y)
    """
    def __init__(
        self, x_dim, y_dim, hidden_dim_list, z1_dim, z2_dim, library_mean, library_logvar, y_prior
    ):
        super().__init__()
        self.save_hyperparameters()

        self.x_to_z1_l = x_to_z1_l(x_dim=x_dim, hidden_dim_list=hidden_dim_list, z1_dim=z1_dim)
        self.z1_l_to_x = z1_l_to_x(z1_dim=z1_dim, hidden_dim_list=hidden_dim_list, x_dim=x_dim)
        self.z1_to_y = MLP(z1_dim, hidden_dim_list, y_dim)
        self.z1_y_to_z2 = GaussianMLP(
            input_dim=z1_dim + y_dim, hidden_dim_list=hidden_dim_list, output_dim=z2_dim
        )
        self.z2_y_to_z1 = GaussianMLP(
            input_dim=z2_dim + y_dim, hidden_dim_list=hidden_dim_list, output_dim=z1_dim
        )

        self.register_buffer("library_mean", torch.from_numpy(np.asarray([library_mean])).float())
        self.register_buffer(
            "library_logvar", torch.from_numpy(np.asarray([library_logvar])).float()
        )
        self.register_buffer("y_prior", torch.from_numpy(np.asarray([y_prior])).float())

    def forward(self, x):
        inference_outputs = self.x_to_z1_l(x=x)
        generative_outputs = self.z1_l_to_x(
            z1_sample=inference_outputs["z1_sample"], l_sample=inference_outputs["l_sample"]
        )
        y_pred_prob = F.softmax(self.z1_to_y(inference_outputs["z1_sample"]), dim=-1)
        return {**inference_outputs, **generative_outputs, "y_pred_prob": y_pred_prob}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.01)

    def mystep(self, batch, stage):
        x, y = batch
        output = self.forward(x=x)
        loss_dict = self.loss(
            x=x,
            y=y,
            px_rate=output["px_rate"],
            px_r=output["px_r"],
            px_dropout=output["px_dropout"],
            l_mu=output["l_mu"],
            l_logvar=output["l_logvar"],
            z1_mu=output["z1_mu"],
            z1_logvar=output["z1_logvar"],
            z1_sample=output["z1_sample"],
        )
        for k, v in loss_dict.items():
            self.log(f"{stage} {k}", v.mean())
        return loss_dict["total loss"].mean()

    def training_step(self, batch, batch_idx):
        return self.mystep(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.mystep(batch, "valid")

    def loss(
        self,
        x,
        y,
        px_rate,
        px_r,
        px_dropout,
        z1_mu,
        z1_logvar,
        z1_sample,
        l_mu,
        l_logvar,
        y_loss_weight=50,
        y_prior_loss_weight=1,
        kl_loss_z1_weight=1,
        kl_loss_z2_weight=1,
        kl_loss_l_weight=1,
    ):
        x_recon_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            .log_prob(x)
            .sum(dim=1)
        )

        y_pred_logit = self.z1_to_y(z1_sample)
        y_loss = F.cross_entropy(y_pred_logit, y, reduction="none")

        y_pred_prob = F.softmax(y_pred_logit)
        y_prior_loss = kl_divergence(
            Categorical(probs=y_pred_prob),
            Categorical(probs=self.y_prior.repeat(y_pred_prob.size(0), 1)),
        )

        kl_loss_z1_q = Normal(z1_mu, torch.exp(0.5 * z1_logvar)).log_prob(z1_sample).sum(dim=1)

        n_labels = y.size()[-1]
        ys, z1_samples = broadcast_labels(z1_sample, n_broadcast=n_labels)
        z2_mus, z2_logvars, z2_samples = self.z1_y_to_z2(x=torch.concat((z1_samples, ys), dim=1))
        pz1_mus, pz1_logvars, _ = self.z2_y_to_z1(x=torch.concat((z2_samples, ys), dim=1))

        kl_loss_z1_p = (
            -Normal(pz1_mus, torch.exp(0.5 * pz1_logvars)).log_prob(z1_samples).sum(dim=1)
        )
        kl_loss_z1_p = (kl_loss_z1_p.view(n_labels, -1).t() * y_pred_prob).sum(dim=-1)
        kl_loss_z1 = kl_loss_z1_p + kl_loss_z1_q

        mean = torch.zeros_like(z2_mus)
        scale = torch.ones_like(z2_logvars)
        kl_loss_z2 = kl_divergence(
            Normal(z2_mus, torch.exp(0.5 * z2_logvars)), Normal(mean, scale)
        ).sum(dim=1)
        kl_loss_z2 = (kl_loss_z2.view(n_labels, -1).t() * y_pred_prob).sum(dim=-1)

        mean = torch.ones_like(l_mu) * self.library_mean
        scale = torch.ones_like(l_logvar) * torch.exp(0.5 * self.library_logvar)
        kl_loss_l = kl_divergence(Normal(l_mu, torch.exp(0.5 * l_logvar)), Normal(mean, scale))

        loss = (
            x_recon_loss
            + y_prior_loss_weight * y_prior_loss
            + kl_loss_z1_weight * kl_loss_z1
            + kl_loss_z2_weight * kl_loss_z2
            + kl_loss_l_weight * kl_loss_l
            + y_loss_weight * y_loss
        )
        loss = torch.mean(loss)

        return {
            "kl_loss_z1_p": kl_loss_z1_p,
            "kl_loss_z1_q": kl_loss_z1_q,
            "kl_loss_z2": kl_loss_z2,
            "kl_loss_l": kl_loss_l,
            "y_loss": y_loss,
            "y_prior_loss": y_prior_loss,
            "x_recon_loss": x_recon_loss,
            "total loss": loss,
        }


class x_to_z1_l(nn.Module):
    def __init__(self, x_dim, hidden_dim_list, z1_dim):
        super().__init__()
        self.x_to_z1 = MLP(x_dim, hidden_dim_list, 2 * z1_dim)
        self.x_to_l = MLP(x_dim, hidden_dim_list, 2)

    def forward(self, x):
        z1_mu, z1_logvar = self.x_to_z1(x).chunk(2, dim=-1)
        z1_mu, z1_logvar, z1_sample = reparameterize(z1_mu, z1_logvar)

        l_mu, l_logvar = self.x_to_l(x).chunk(2, dim=-1)
        l_mu, l_logvar, l_sample = reparameterize(l_mu, l_logvar)
        l_sample = torch.exp(torch.clip(l_sample, max=10))  # clip for numerical stability

        return {
            "l_sample": l_sample,
            "l_mu": l_mu,
            "l_logvar": l_logvar,
            "z1_sample": z1_sample,
            "z1_mu": z1_mu,
            "z1_logvar": z1_logvar,
        }


class z1_l_to_x(nn.Module):
    def __init__(self, z1_dim, hidden_dim_list, x_dim):
        super().__init__()
        # Mean expression per gene, before library-size correction, positive and sum to one
        self.z1_to_scale = nn.Sequential(
            MLP(z1_dim, hidden_dim_list, x_dim),
            nn.Softmax(),
        )

        # Logits for Dropout, free range
        self.z1_to_h = MLP(z1_dim, hidden_dim_list, x_dim)

        # Gene-wise dispersion parameter in ZINB, to be made positive later by exponentiation
        self.theta = nn.Parameter(torch.randn(x_dim))

    def forward(self, z1_sample, l_sample):
        px_scale = self.z1_to_scale(z1_sample)
        px_dropout = self.z1_to_h(z1_sample)
        px_rate = torch.clip(l_sample * px_scale, max=1e6)
        px_r = torch.exp(self.theta)
        return {"px_rate": px_rate, "px_r": px_r, "px_dropout": px_dropout}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, output_dim):
        super().__init__()
        layers = []
        in_dim = input_dim

        for dim in hidden_dim_list:
            layers += [
                nn.Linear(in_dim, dim),
                nn.ReLU(),
            ]
            in_dim = dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, output_dim):
        super().__init__()
        self.network = MLP(input_dim, hidden_dim_list, 2 * output_dim)

    def forward(self, x):
        mu, logvar = self.network(x).chunk(2, dim=-1)
        mu, logvar, sample = reparameterize(mu, logvar)
        return mu, logvar, sample

import torch
import torch.nn as nn

class JacobianRegulariser:
    """
    Reference: https://arxiv.org/abs/1908.02729
    """
    def __init__(self, model: nn.Module, num_projections: int = 1):
        self.model = model
        self.num_projections = num_projections
        self.device = next(model.parameters()).device

    def _get_random_projections(self, batch_size, output_dim):
        v = torch.randn(self.num_projections, batch_size, output_dim, device=self.device)
        return v / torch.clamp(v.norm(dim=-1, keepdim=True), min=1e-8)

    def compute_jacobian_norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True).to(self.device)

        output = self.model(x)
        batch_size, output_dim = output.shape

        v_vectors = self._get_random_projections(batch_size, output_dim)

        norm_sum = torch.zeros(batch_size, device=self.device)

        for i in range(self.num_projections):
            v = v_vectors[i]
            Jv = torch.autograd.grad(
                outputs=output,
                inputs=x,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True,
            )[0]

            Jv_flat = Jv.flatten(start_dim=1)
            norm_sum += (Jv_flat ** 2).sum(dim=1)

        scaling = output_dim / self.num_projections
        return torch.sqrt(torch.clamp(norm_sum * scaling, min=1e-8))

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.compute_jacobian_norm(inputs).mean()

class Centroids:
    def __init__(self, model: nn.Module, device=None):
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device

    def centroids(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True).to(self.device)
        output = self.model(x)
        grad_output = torch.ones_like(output)
        centroids = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=grad_output
        )[0]
        return centroids
    
    def inner_product(self, x:torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True).to(self.device)
        centroids = self.compute_centroids(x)
        centroids = centroids.reshape(centroids.size(0),-1)
        x = x.reshape(x.size(0),-1)
        inner_products = (centroids * x).sum(dim=1)
        return inner_products

    def alignments(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True).to(self.device)
        centroids = self.compute_centroids(x)
        centroids = centroids.reshape(centroids.size(0),-1)
        x = x.reshape(x.size(0),-1)
        sims = (centroids * x).sum(dim=1) / torch.clamp(centroids.norm(dim=1) * x.norm(dim=1), min=1e-8)
        return sims

    def norms(self, x: torch.Tensor) -> torch.Tensor:
        centroids = self.compute_centroids(x)
        return centroids.norm(dim=1)
    
class PC1:
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device

    def _compute_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True).to(self.device)
        output = self.model(x)
        output_dim = output.shape[1] if output.ndim == 2 else 1
        batch_size = x.size(0)

        jacobians = []

        for i in range(output_dim):
            grad_output = torch.zeros_like(output)
            if output_dim == 1:
                grad_output[:] = 1.0
            else:
                grad_output[:, i] = 1.0

            grad = torch.autograd.grad(
                outputs=output,
                inputs=x,
                grad_outputs=grad_output,
            )[0]

            jacobians.append(grad.flatten(start_dim=1))

        return torch.stack(jacobians, dim=1) if output_dim > 1 else jacobians[0].unsqueeze(1)

    def _explained_variance_first_pc(self, jac: torch.Tensor) -> torch.Tensor:
        B, C, D = jac.shape
        jac_flat = jac.reshape(B * C, D)

        jac_flat = jac_flat - jac_flat.mean(dim=0, keepdim=True)

        u, s, v = torch.svd(jac_flat, some=False)
        explained_var = s[0]**2 / torch.clamp(s.pow(2).sum(), min=1e-8)
        return explained_var

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        jac = self._compute_jacobian(x)
        return self._explained_variance_first_pc(jac)
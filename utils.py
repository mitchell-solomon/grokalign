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

    def compute_centroids(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True).to(self.device)
        output = self.model(x)
        grad_output = torch.ones_like(output)
        centroids = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=grad_output,
            retain_graph=True
        )[0]
        return centroids
    
    def compute_inner_product(self, x:torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True).to(self.device)
        centroids = self.compute_centroids(x)
        centroids = centroids.reshape(centroids.size(0),-1)
        x = x.reshape(x.size(0),-1)
        inner_products = (centroids * x).sum(dim=1)
        return inner_products

    def compute_alignments(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True).to(self.device)
        centroids = self.compute_centroids(x)
        centroids = centroids.reshape(centroids.size(0),-1)
        x = x.reshape(x.size(0),-1)
        sims = (centroids * x).sum(dim=1) / torch.clamp(centroids.norm(dim=1) * x.norm(dim=1), min=1e-8)
        return sims

    def compute_norms(self, x: torch.Tensor) -> torch.Tensor:
        centroids = self.compute_centroids(x)
        return centroids.norm(dim=1)
    
class PC1:
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device

    def _compute_pc1s(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True).to(self.device)
        pc1s=[]
        for k in range(x.size()[0]):
            J=torch.autograd.functional.jacobian(self.model,x[k:k+1])[0,:,0]
            J=J.reshape(J.size()[0],-1)
            U, S, Vh = torch.linalg.svd(J, full_matrices=False)
            S_squared = S ** 2
            explained_var = S_squared[0] / S_squared.sum()
            pc1s.append(explained_var.item())
        pc1s=torch.tensor(pc1s)
        return pc1s

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pc1s = self._compute_pc1s(x)
        return pc1s.mean().item()
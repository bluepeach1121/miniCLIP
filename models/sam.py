import torch
from torch.optim import Optimizer, AdamW

class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) Optimizer (one-step), adapted for mixed-precision.
    Typical usage (pseudocode):
        scaler = GradScaler()
        for images, texts in dataloader:
            # 1) forward-backward pass
            with autocast():
                loss = compute_loss(...)
            scaler.scale(loss).backward()

            # 2) first_step
            optimizer.first_step(scaler=scaler, zero_grad=True)

            # 3) forward-backward pass again
            with autocast():
                loss2 = compute_loss(...)
            scaler.scale(loss2).backward()

            # 4) second_step
            optimizer.second_step(scaler=scaler, zero_grad=True)

            # 5) update scaler
            scaler.update()
    """
    def __init__(self, params, base_optimizer=AdamW, rho=0.05, epsilon=1e-12, **kwargs):
        if rho < 0:
            raise ValueError(f"Invalid rho, must be non-negative: {rho}")

        defaults = dict(rho=rho, epsilon=epsilon, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups)

    @torch.no_grad()
    def first_step(self, scaler=None, zero_grad=False):
        """
        1) Unscale gradients (if using GradScaler).
        2) Compute per-layer gradient norm & ascend in that direction by rho.
        3) Store the perturbations (e_w) so we can revert them in second_step().
        4) Optionally zero grad.
        """
        if scaler is not None:
            scaler.unscale_(self.base_optimizer)

        for group in self.param_groups:
            scale = group["rho"]
            eps   = group["epsilon"]
            group["_e_ws"] = []   

            layer_sq_sum = 0.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                layer_sq_sum += g.square().sum().item()

            grad_norm = max(layer_sq_sum**0.5, eps)

            for p in group["params"]:
                if p.grad is None:
                    group["_e_ws"].append(None)
                    continue

                e_w = (scale / grad_norm) * p.grad
                p.add_(e_w) 
                group["_e_ws"].append(e_w)  

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, scaler=None, zero_grad=False):
        """
        1) Revert the weights using the stored e_w from first_step.
        2) Take the actual optimizer step with GradScaler (if any).
        3) Optionally zero grad.
        """
        for group in self.param_groups:
            e_ws = group["_e_ws"]
            for p, e_w in zip(group["params"], e_ws):
                if (p.grad is None) or (e_w is None):
                    continue
                p.sub_(e_w)  

        if scaler is not None:
            scaler.step(self.base_optimizer)
        else:
            self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def zero_grad(self):
        super().zero_grad()
        self.base_optimizer.zero_grad()

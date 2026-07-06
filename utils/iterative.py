import torch
from utils.general import true_logits, forward_batched

############################################################################################
# iterative

def prox_l0_ball(y: torch.Tensor, n_nonzero: int):
    """Project onto the L0 ball: keep the n_nonzero largest-magnitude entries."""
    _, top_idx = y.flatten().abs().topk(n_nonzero)
    xbar = torch.zeros_like(y)
    xbar.flatten()[top_idx] = y.flatten()[top_idx]
    return xbar, 0.0

def scale_value(s, act_diff, model, hook_names, batches, n_samples):
    """Compute the negative true output logits depending on the scaling factor s."""
    # compute interventions
    intervention = s * act_diff
    intervention_dict = {hook_names[l]: intervention[l] for l in range(model.cfg.n_layers)}
    # add hooks
    def intervention_hook(act, hook):
        act = torch.cat(
            [act[:, :-1], (act[:, -1] + intervention_dict[hook.name]).unsqueeze(1)],
            dim=1)
        return act
    model.reset_hooks()
    for h in hook_names:
        model.add_hook(h, intervention_hook, "fwd")
    # foward passes over all batches
    total_value = torch.tensor(0.0, device=torch.device('cuda'))
    for tokens, out_indices_true in batches:
        value = true_logits(forward_batched(model, tokens), out_indices_true)
        total_value -= value
    total_value = total_value / n_samples
    # cleanup
    model.reset_hooks()
    return total_value.item()

def scale_grad(s, act_diff, model, hook_names, batches, n_samples):
    """Compute the gradient of the negative true output logits w.r.t. the scaling factor s."""
    # reset gradients of s
    s = s.detach().requires_grad_(True)
    # compute interventions
    intervention = s * act_diff
    intervention_dict = {hook_names[l]: intervention[l] for l in range(model.cfg.n_layers)}
    # add hooks
    def intervention_hook(act, hook):
        act = torch.cat(
            [act[:, :-1], (act[:, -1] + intervention_dict[hook.name]).unsqueeze(1)],
            dim=1)
        return act
    model.reset_hooks()
    for h in hook_names:
        model.add_hook(h, intervention_hook, "fwd")
    # foward passes over all batches
    total_value = torch.tensor(0.0, device=torch.device('cuda'))
    for tokens, out_indices_true in batches:
        value = true_logits(forward_batched(model, tokens), out_indices_true)
        total_value -= value
    total_value = total_value / n_samples
    # gradient computation
    total_value.backward()
    grad = s.grad
    # cleanup
    model.reset_hooks()
    return total_value.item(), grad.detach()
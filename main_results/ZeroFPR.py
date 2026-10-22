"""
ZeroFPR: Proximal gradient algorithm with quasi-Newton acceleration.

Solves:
    minimize  f(x) + g(x)

where f is smooth (gradient available) and g is proximable (e.g. indicator of
an L0 ball, which enforces sparsity via hard thresholding).

Reference:
    Themelis, Stella, Patrinos (2018). "Forward-Backward Envelope for the Sum
    of Two Nonconvex Functions: Further Properties and Nonmonotone Line-Search
    Algorithms." SIAM J. Optim., 28(3):2274-2303.
    https://epubs.siam.org/doi/10.1137/16M1080240

Julia reference implementation:
    https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl
"""

import torch
import math

# ---------------------------------------------------------------------------
# L-BFGS direction
# ---------------------------------------------------------------------------

class LBFGS:
    """Two-loop L-BFGS inverse Hessian approximation.

    Stores (s, y) pairs where:
        s = xbar_k - xbar_{k-1}   (step in solution space)
        y = res_k  - res_{k-1}    (fixed-point residual difference)
    """

    def __init__(self, memory: int = 5):
        self.memory = memory
        self.s_list = []
        self.y_list = []
        self.H0 = 1.0  # initial Hessian scaling

    def reset(self):
        self.s_list.clear()
        self.y_list.clear()
        self.H0 = 1.0

    def update(self, s: torch.Tensor, y: torch.Tensor):
        sy = torch.dot(s.flatten(), y.flatten()).item()
        if sy <= 0:
            return  # skip: curvature condition not satisfied
        self.H0 = sy / torch.dot(y.flatten(), y.flatten()).item()
        self.s_list.append(s.clone())
        self.y_list.append(y.clone())
        if len(self.s_list) > self.memory:
            self.s_list.pop(0)
            self.y_list.pop(0)

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        """Compute H @ r via the two-loop recursion."""
        q = r.clone().flatten()
        alphas = []
        rhos = []
        for s, y in zip(reversed(self.s_list), reversed(self.y_list)):
            rho = 1.0 / torch.dot(y.flatten(), s.flatten()).item()
            alpha = rho * torch.dot(s.flatten(), q).item()
            q = q - alpha * y.flatten()
            alphas.append(alpha)
            rhos.append(rho)
        r_out = self.H0 * q
        for s, y, alpha, rho in zip(self.s_list, self.y_list,
                                     reversed(alphas), reversed(rhos)):
            beta = rho * torch.dot(y.flatten(), r_out).item()
            r_out = r_out + (alpha - beta) * s.flatten()
        return r_out.view_as(r)


# ---------------------------------------------------------------------------
# FBE and model value helpers
# ---------------------------------------------------------------------------

def _f_model(f_x: float, grad_f_x: torch.Tensor, res: torch.Tensor,
             L: float) -> float:
    """Upper bound model: f(x) - <grad_f_x, res> + (L/2)*||res||^2."""
    return (f_x
            - torch.dot(grad_f_x.flatten(), res.flatten()).item()
            + 0.5 * L * res.flatten().dot(res.flatten()).item())


def _fbe(f_x: float, grad_f_x: torch.Tensor, res: torch.Tensor,
         g_xbar: float, alpha: float, gamma: float) -> float:
    return _f_model(f_x, grad_f_x, res, alpha / gamma) + g_xbar

# ---------------------------------------------------------------------------
# ZeroFPR main function
# ---------------------------------------------------------------------------

def ZeroFPR(
    x0: torch.Tensor,
    f,
    f_and_grad,          # callable: x -> (value: float, grad: Tensor)
    prox_g,              # callable: (y, gamma) -> (xbar: Tensor, g_val: float)
    alpha: float = 0.95,
    beta: float = 0.5,
    min_gamma: float = 1e-7,
    memory: int = 5,
    max_iter: int = 500,
    max_backtracks: int = 20,
    tol: float = 1e-8,
    verbose: bool = False,
    initial_overshoot: bool = True,
    probe_mask: torch.Tensor = None,
):
    """
    ZeroFPR algorithm.

    Args:
        x0:         Initial point.
        f:          Returns f(x) for a given x.
        f_and_grad: Returns (f(x), grad_f(x)) for a given x.
        prox_g:     Returns (prox_{gamma*g}(y), g(prox)) for given (y, gamma).
        alpha:      Step size safety factor (0 < alpha < 1).
        beta:       Line search backtrack factor (0 < beta < 1).
        min_gamma:  Lower bound on gamma.
        memory:     L-BFGS memory size.
        max_iter:   Maximum number of outer iterations.
        max_backtracks: Maximum line search halvings per iteration.
        tol:            Convergence tolerance on ||res||_inf / gamma.
        verbose:        Print progress.
        initial_overshoot: Take an oversized (1.5*gamma) prox-gradient pre-step.
        probe_mask:     Optional boolean mask confining the Lipschitz step.

    Returns:
        xbar:          Solution (proximal point at final iterate).
        trace:         List of FBE values per iteration.
    """
    x = x0.clone()
    f_x, grad_f_x = f_and_grad(x)

    # --- initialize gamma ---
    # estimate Lipschitz constant via finite difference
    x_eps = x + 1.0 if probe_mask is None else x + probe_mask
    _, grad_eps = f_and_grad(x_eps)
    grad_diff = grad_eps - grad_f_x
    if probe_mask is not None:
        grad_diff = grad_diff * probe_mask
    n_probe = x.numel() if probe_mask is None else int(probe_mask.sum().item())
    Lf_est = torch.linalg.vector_norm(grad_diff).item() / math.sqrt(n_probe)
    Lf_est = max(Lf_est, 1e-6)
    gamma = alpha / Lf_est

    # --- initial gradient step ---
    # skipped for the relevance-based warm start (enters the loop from x0 as-is)
    if initial_overshoot:
        x, _ = prox_g(x - 1.5*gamma*grad_f_x, gamma)
        f_x, grad_f_x = f_and_grad(x)

    # --- initial forward-backward step ---
    y = x - gamma * grad_f_x
    xbar, g_xbar = prox_g(y, gamma)
    res = x - xbar

    lbfgs = LBFGS(memory=memory)

    xbar_prev = None
    res_xbar_prev = None

    trace = []
    eps = torch.finfo(x.dtype).eps

    for j in range(max_iter):

        # ------------------------------------------------------------------
        # Step 1: adaptive gamma
        # ------------------------------------------------------------------
        prev_gamma = gamma
        f_xbar = f(xbar)
        f_upp = _f_model(f_x, grad_f_x, res, alpha / gamma)
        btol = 10 * eps * (1 + abs(f_xbar))
        while (gamma >= min_gamma) and (f_xbar > f_upp + btol):
            gamma *= 0.5
            y = x - gamma * grad_f_x
            xbar, g_xbar = prox_g(y, gamma)
            res = x - xbar
            f_xbar = f(xbar)
            f_upp = _f_model(f_x, grad_f_x, res, alpha / gamma)
            btol = 10 * eps * (1 + abs(f_xbar))
        if gamma != prev_gamma:
            lbfgs.reset()

        # ------------------------------------------------------------------
        # Step 2: FBE at current x
        # ------------------------------------------------------------------
        FBE_x = f_upp + g_xbar

        # ------------------------------------------------------------------
        # Step 3: gradient of f at xbar, residual at xbar
        # ------------------------------------------------------------------
        f_xbar, grad_f_xbar = f_and_grad(xbar)
        y_bar = xbar - gamma * grad_f_xbar
        xbarbar, _ = prox_g(y_bar, gamma)
        res_xbar = xbar - xbarbar

        # ------------------------------------------------------------------
        # Step 4: L-BFGS update and direction
        # ------------------------------------------------------------------
        if xbar_prev is not None:
            s = xbar - xbar_prev
            y_lbfgs = res_xbar - res_xbar_prev
            lbfgs.update(s, y_lbfgs)

        if len(lbfgs.s_list) > 0:
            d = -lbfgs.apply(res_xbar)
        else:
            d = -res_xbar  # fallback: gradient step direction

        # ------------------------------------------------------------------
        # Step 5: line search over FBE
        # ------------------------------------------------------------------
        tau = 1.0
        sigma = beta * (0.5 / gamma) * (1 - alpha)
        tol_ls = 10 * eps * (1 + abs(FBE_x))
        threshold = FBE_x - sigma * res.flatten().dot(res.flatten()).item() + tol_ls

        x_new = x
        f_x_new, grad_f_x_new = f_x, grad_f_x
        xbar_new, g_xbar_new = xbar, g_xbar
        res_new = res

        for k in range(max_backtracks):

            x_new = xbar + tau * d
            f_x_new, grad_f_x_new = f_and_grad(x_new)
            y_new = x_new - gamma * grad_f_x_new
            xbar_new, g_xbar_new = prox_g(y_new, gamma)
            res_new = x_new - xbar_new

            FBE_new = _fbe(f_x_new, grad_f_x_new, res_new, g_xbar_new, alpha, gamma)
            if FBE_new <= threshold:
                break

            tau = 0.5*tau if k < max_backtracks - 2 else 0.0

        # ------------------------------------------------------------------
        # Step 6: accept step, save state
        # ------------------------------------------------------------------
        xbar_prev = xbar.clone()
        res_xbar_prev = res_xbar.clone()

        x = x_new
        f_x = f_x_new
        grad_f_x = grad_f_x_new
        xbar = xbar_new
        g_xbar = g_xbar_new
        res = res_new

        # ------------------------------------------------------------------
        # Step 7: convergence check
        # ------------------------------------------------------------------
        trace.append(FBE_new)
        norm_res = res.flatten().abs().max().item()
        if verbose:
            print(f"iter {j:4d}  FBE={FBE_new:.6f}  ||res||_inf/gamma={norm_res/gamma:.2e}")
        if norm_res / gamma <= tol:
            break

    return xbar, trace

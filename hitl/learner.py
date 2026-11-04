"""The learner: known linear model + small neural residual kept orthogonal to the known terms,
    edge cost f(e) = c' d_e + g(e),   g(e) = r(z_e) - a(omega)' d_e,   r(z) = b' tanh(W1' z + b1)
    (z = the attributes visible to the learner, d = its known terms; the h entries of tanh(.) are the learned features).
a(omega) is the least-squares fit of the network's changes on the known terms' changes over a bank of small feasible
reassignments (the ruler), so on plan differences the residual g has no component along any known term: reweighting
what is known is c's job, g only carries what the known terms cannot express.  The network still sees every raw
attribute (thresholds and interactions need them).
Training target: the observed response x^H should be the model's best feasible plan, by a margin (the learner does
not know the human's response model).  Structured hinge, refit on every past month (exact loss-augmented search over
all feasible plans):
    loss = max_{y feasible} [ mu * d(x^H, y) + f(x^H) - f(y) ]_+ / n,   d = cargoes assigned differently
and, when the human chose q among several shown plans y (use_choice), for each of them
    [ mu * d(q, y) + f(q) - f(y) ]_+ / n.
plus lam_wd * |network weights|^2.  Adam; after every step the model is rescaled to the truth's cost spread over real
pairs (decisions unchanged), the scale-shrinking direction is removed from each loss step, and no learned-feature
weight may exceed 5x the truth's spread.  The oracle baseline is the same class with every true column linear and no
network."""
import numpy as np

from .assign import solve


BMAX = 5.0          # largest weight of one learned feature, in units of the truth's cost spread


class Model:
    def __init__(self, lin_cols, c0, n_hidden=0, rng=None, zcols=(), proj=None):
        """proj = (Dk, R): known-term changes (M x |cols|) and the network inputs of the same M ruler moves."""
        self.cols = np.asarray(lin_cols); self.zc = np.asarray(zcols, dtype=int); self.p = {"c": np.asarray(c0, float).copy()}
        self.nn = n_hidden > 0 and len(self.zc) > 0; nz = len(self.zc)
        if self.nn:
            self.p.update(W1=rng.standard_normal((nz, n_hidden)) / np.sqrt(nz), b1=np.zeros(n_hidden),
                          b=rng.standard_normal(n_hidden) * 1e-3)
        self.proj = None
        if self.nn and proj is not None:
            Dk, R = proj; self.proj = (np.linalg.pinv(Dk), Dk, R)
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}; self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    # ---- evaluation
    def residual(self, Z):
        if not self.nn:
            return np.zeros(Z.shape[:-1])
        return np.tanh(Z @ self.p["W1"] + self.p["b1"]) @ self.p["b"]

    def ruler_changes(self, R):
        """Raw network changes over ruler moves R = (Zo, Zn, Zm)."""
        Zo, Zn, Zm = R
        return ((self.residual(Zn[..., self.zc]) - self.residual(Zo[..., self.zc])) * Zm).sum(1)

    def ahat(self):
        """The part of the network that is just a reweighting of the known terms (on plan differences)."""
        if self.proj is None:
            return None
        A, Dk, R = self.proj
        return A @ self.ruler_changes(R)

    def residual_ruler(self):
        """Changes of the orthogonal residual g over the projection's ruler moves."""
        A, Dk, R = self.proj
        dr = self.ruler_changes(R)
        return dr - Dk @ (A @ dr)

    def costs(self, X):
        f = X[..., self.cols] @ self.p["c"]
        if self.nn:
            f = f + self.residual(X[..., self.zc])
            a = self.ahat()
            if a is not None:
                f = f - X[..., self.cols] @ a
        return f

    # ---- gradient of sum_e w_e f(e)
    def _nn_grad(self, Z, wts):
        H = np.tanh(Z @ self.p["W1"] + self.p["b1"]); u = self.p["b"]
        dA = (wts[:, None] * u[None, :]) * (1 - H ** 2)
        return {"b": wts @ H, "W1": Z.T @ dA, "b1": dA.sum(0)}

    def grad_edges(self, Xl, Z, wts):
        s = wts @ Xl; g = {"c": s}
        if self.nn:
            g.update(self._nn_grad(Z, wts))
            if self.proj is not None:                 # minus the gradient of s' a(omega) = (A' s)' (ruler changes)
                A, Dk, (Zo, Zn, Zm) = self.proj; v = A.T @ s; wn = (v[:, None] * Zm).ravel(); nz = len(self.zc)
                g1 = self._nn_grad(Zn[..., self.zc].reshape(-1, nz), wn); g2 = self._nn_grad(Zo[..., self.zc].reshape(-1, nz), -wn)
                for k in g1:
                    g[k] = g[k] - g1[k] - g2[k]
        return g

    def adam(self, g, lr, b1=0.9, b2=0.999):
        self.t += 1
        for k, gk in g.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk; self.v[k] = b2 * self.v[k] + (1 - b2) * gk ** 2
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] = self.p[k] - lr * mh / (np.sqrt(vh) + 1e-8)

    def rescale(self, E):
        """Scale so the model's costs have the truth's spread over real pairs (each cargo's own level removed);
        decisions unchanged."""
        Xe, ge, target = E
        f = self.costs(Xe); mu = np.bincount(ge, f) / np.bincount(ge)
        s = float(np.sqrt(np.mean((f - mu[ge]) ** 2))) / target
        if s > 1e-12:
            self.p["c"] = self.p["c"] / s
            if self.nn:
                self.p["b"] = self.p["b"] / s

    # ---- one refit pass over the history
    def refit(self, history, n, lr, epochs, lam_wd, E, rng, margin=0.0):
        for _ in range(epochs):
            for s in rng.permutation(len(history)):
                mo, q, xh, others = history[s]; Cm = self.costs(mo.X); Ca = Cm.copy(); Ca[np.arange(n), xh] += margin
                xt = solve(Ca, mo.allowed)
                hinge = margin * np.sum(xt != xh) + Cm[np.arange(n), xh].sum() - Cm[np.arange(n), xt].sum()
                g = {k: np.zeros_like(v) for k, v in self.p.items()}
                if hinge > 1e-12 and not np.array_equal(xt, xh):
                    idx_h = (np.arange(n), xh); idx_t = (np.arange(n), xt)
                    Xl = np.concatenate([mo.X[idx_h][:, self.cols], mo.X[idx_t][:, self.cols]])
                    Z = np.concatenate([mo.X[idx_h][:, self.zc], mo.X[idx_t][:, self.zc]])
                    wts = np.r_[np.ones(n), -np.ones(n)] / n
                    gl = self.grad_edges(Xl, Z, wts)
                    if self.nn:      # the loss can always be lowered by shrinking the whole model; remove that direction
                        rad = np.r_[self.p["c"], self.p["b"]]; gr = np.r_[gl["c"], gl["b"]]
                        pr = (gr @ rad) / max(rad @ rad, 1e-12); gl["c"] = gl["c"] - pr * self.p["c"]; gl["b"] = gl["b"] - pr * self.p["b"]
                    for k, v in gl.items():
                        g[k] += v
                for y in others:     # the human chose q over y: q should beat y by the margin
                    hy = margin * np.sum(y != q) + Cm[np.arange(n), q].sum() - Cm[np.arange(n), y].sum()
                    if hy > 1e-12:
                        iq = (np.arange(n), q); iy = (np.arange(n), y)
                        gl = self.grad_edges(np.concatenate([mo.X[iq][:, self.cols], mo.X[iy][:, self.cols]]),
                                             np.concatenate([mo.X[iq][:, self.zc], mo.X[iy][:, self.zc]]),
                                             np.r_[np.ones(n), -np.ones(n)] / n)
                        for k, v in gl.items():
                            g[k] += v
                if self.nn:
                    for k in ("W1", "b"):
                        g[k] += 2 * lam_wd * self.p[k]
                self.adam(g, lr); self.rescale(E)
                if self.nn:          # max-norm: no single learned feature may dominate (features lie in [-1, 1])
                    self.p["b"] = np.clip(self.p["b"], -BMAX * E[2], BMAX * E[2])

"""The machine: one linear weight per feature, cost of a pair = c' x.
Training target: the human's plan x^H should be the best feasible plan by a margin (structured hinge, exact
loss-augmented search over all feasible plans), refit on every past month:
    loss = max_y [ mu * d(x^H, y) + f(x^H) - f(y) ]_+ / n,   d = cargoes assigned differently.
Adam; after every step the weights are rescaled to the truth's cost spread over real pairs (no decision changes)."""
import numpy as np

from .assign import solve


class Linear:
    def __init__(self, c0, W):
        self.c = np.asarray(c0, float).copy(); self.W = W; self.m = np.zeros_like(self.c); self.v = np.zeros_like(self.c); self.t = 0
        self.rescale()

    def costs(self, mo):
        return mo.X @ self.c

    def rescale(self):
        Xe, ge = self.W.base.Xe, self.W.base.ge
        f = Xe @ self.c; mu = np.bincount(ge, f) / np.bincount(ge)
        s = float(np.sqrt(np.mean((f - mu[ge]) ** 2))) / self.W.target
        if s > 1e-12:
            self.c /= s

    def adam(self, g, lr, b1=0.9, b2=0.999):
        self.t += 1; self.m = b1 * self.m + (1 - b1) * g; self.v = b2 * self.v + (1 - b2) * g * g
        self.c -= lr * (self.m / (1 - b1 ** self.t)) / (np.sqrt(self.v / (1 - b2 ** self.t)) + 1e-8)

    def refit(self, history, lr, epochs, margin, rng):
        for _ in range(epochs):
            for s in rng.permutation(len(history)):
                mo, xh = history[s]; n = mo.n; idx = np.arange(n)
                Cm = self.costs(mo); Ca = Cm.copy(); Ca[idx, xh] += margin; xt = solve(Ca, mo.allowed)
                hinge = margin * np.sum(xt != xh) + Cm[idx, xh].sum() - Cm[idx, xt].sum()
                g = (mo.X[idx, xh].sum(0) - mo.X[idx, xt].sum(0)) / n if hinge > 1e-12 and np.any(xt != xh) else np.zeros_like(self.c)
                self.adam(g, lr); self.rescale()

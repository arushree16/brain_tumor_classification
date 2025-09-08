import tensorflow as tf

class SAM:
    """Sharpness-Aware Minimization (SAM) wrapper for custom training loops."""
    def __init__(self, base_optimizer, rho=0.05, adaptive=False):
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive

    def first_step(self, grads, vars):
        grad_norm = tf.linalg.global_norm([g for g in grads if g is not None])
        scale = self.rho / (grad_norm + 1e-12)

        e_ws = []
        for g, v in zip(grads, vars):
            if g is None:
                e_ws.append(tf.zeros_like(v))
                continue

            if self.adaptive:
                # Adaptive SAM: scale by |w| instead of w² for better stability
                e_w = (tf.abs(v) * g) * scale
            else:
                e_w = g * scale

            v.assign_add(e_w)
            e_ws.append(e_w)

        return e_ws

    def second_step(self, grads, vars, e_ws):
        for v, e_w in zip(vars, e_ws):
            v.assign_sub(e_w)
        self.base_optimizer.apply_gradients(zip(grads, vars))

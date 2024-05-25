# Author: Bingxin Ke
# Last modified: 2024-02-22

import numpy as np


class IterExponential:
    def __init__(self, total_iter_length, final_ratio, warmup_steps=0) -> None:
        """
        Customized iteration-wise exponential scheduler.
        Re-calculate for every step, to reduce error accumulation

        Args:
            total_iter_length (int): Expected total iteration number
            final_ratio (float): Expected LR ratio at n_iter = total_iter_length
        """
        self.total_length = total_iter_length
        self.effective_length = total_iter_length - warmup_steps
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps

    def __call__(self, n_iter) -> float:
        if n_iter < self.warmup_steps:
            alpha = 1.0 * n_iter / self.warmup_steps
        elif n_iter >= self.total_length:
            alpha = self.final_ratio
        else:
            actual_iter = n_iter - self.warmup_steps
            alpha = np.exp(
                actual_iter / self.effective_length * np.log(self.final_ratio)
            )
        return alpha


if "__main__" == __name__:
    lr_scheduler = IterExponential(
        total_iter_length=50000, final_ratio=0.01, warmup_steps=200
    )
    lr_scheduler = IterExponential(
        total_iter_length=50000, final_ratio=0.01, warmup_steps=0
    )

    x = np.arange(100000)
    alphas = [lr_scheduler(i) for i in x]
    import matplotlib.pyplot as plt

    plt.plot(alphas)
    plt.savefig("lr_scheduler.png")

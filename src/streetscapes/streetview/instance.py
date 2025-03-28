import numpy as np

import matplotlib.pyplot as plt

class SVInstance:
    def __init__(self, mask: np.ma.MaskedArray):
        self.mask = mask

    def __repr__(self):
        return f"Instance(mask={self.mask})"

    def plot(self):
        plt.figure()
        plt.imshow(self.mask)

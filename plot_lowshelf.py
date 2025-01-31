from Filters import *
import torch

import matplotlib.pyplot as plt 

frequencies = torch.linspace(20,20e3, 1024)

CUT_OFF = torch.tensor(1000.0)

GAIN = torch.tensor(3.0) #dB

low_shelf = RBJ_Bell(frequencies, CUT_OFF, GAIN, q_factor=0.7071)

plt.semilogx(frequencies, 20.0 * torch.log10(low_shelf.response), color='red')
plt.savefig("shelf.png")
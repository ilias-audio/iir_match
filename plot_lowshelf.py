from Filters import *
import torch
import scipy.signal as signal
import matplotlib.pyplot as plt 
import numpy as np


NFFT = 2**13
SAMPLE_RATE= 48e3
frequencies = torch.linspace(2,SAMPLE_RATE/2, NFFT)

CUT_OFF = torch.tensor(10000.0)

GAIN = torch.tensor(-24.0) #dB

low_shelf = RBJ_HighShelf(frequencies, CUT_OFF, GAIN, q_factor=0.5, sample_rate=SAMPLE_RATE)

low_shelf.compute_sos()

print(low_shelf.sos.shape)

plt.semilogx(frequencies, 20.0 * torch.log10(low_shelf.response), color='red')


w, h = signal.sosfreqz(low_shelf.sos.numpy(), worN=NFFT, fs=SAMPLE_RATE)

plt.semilogx(w, 20.0 * np.log10(abs(h)), color='blue')


plt.savefig("shelf.png")
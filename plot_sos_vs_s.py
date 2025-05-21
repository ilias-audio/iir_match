import torch 
import Filters
import matplotlib.pyplot as plt
import os
# parameters vectors
N = 512

cutoff_freqs = torch.tensor([200., 3000., 8000.]).unsqueeze(-1)
gains        = torch.tensor([-6., 12., -20.]).unsqueeze(-1)
Q            = torch.tensor([1., 2., .5]).unsqueeze(-1)


freqency_axis = torch.logspace(torch.log10(torch.tensor(1)), torch.log10(torch.tensor(20000)), N)

s_plane_response = Filters.evaluate_mag_response(freqency_axis, cutoff_freqs, gains, Q)
sos_response = Filters.evaluate_sos_response(freqency_axis, cutoff_freqs, gains, Q)

fftfreqs = torch.linspace(0, 24e3, N)

plt.semilogx(freqency_axis, 20. * torch.log10(s_plane_response))
plt.semilogx(fftfreqs, 20. * torch.log10(abs(sos_response)))

plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), "s_plane.png"))
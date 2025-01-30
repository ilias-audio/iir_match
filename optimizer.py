import torch
from tqdm import tqdm
import numpy as np
from Filters import *

###############################################################################
# UTILITIES
###############################################################################

def log_normalize(value, min, max):
  value = (torch.log(value) - torch.log(min)) / (torch.log(max) - torch.log((min)))
  return (value)

def log_denormalize(value, min, max):
  value = 10 ** (value * torch.log10(max) + (1.0 - value) * torch.log10(min))
  return value


def lin_normalize(value, min, max):
  value = ((value) * (max - min)) + min
  return value

def lin_denormalize(value, min, max):
  value = min + (max - min) * value
  return value

def frequency_denormalize(f):
  min = torch.tensor(20.0)
  max = torch.tensor(20000.0)

  f = log_denormalize(f, min, max)
  return f

def frequency_normalize(f):
  min = torch.tensor(20.0)
  max = torch.tensor(20000.0)

  f = log_normalize(f, min, max)
  return f

def gain_denormalize(g):
  min = torch.tensor(-12.0)
  max = torch.tensor(12.0)

  real_gain = lin_denormalize(g, min, max)

  g = 10.0 ** (real_gain/40.0)

  return g

def gain_normalize(g):
  min = torch.tensor(-12.0)
  max = torch.tensor(12.0)

  g = lin_normalize(g, min, max)

  return g

def q_denormalize(q):

  min = torch.tensor(0.01)
  max = torch.tensor(4.0)

  return log_denormalize(q, min, max)

def q_normalize(q):

  min = torch.tensor(0.01)
  max = torch.tensor(4.0)

  return log_normalize(q, min, max)

def evaluate_mag_response(
    x: torch.Tensor,     # Frenquency vector
    F: torch.Tensor,     # Center frequencies
    G: torch.Tensor,     # Gain values
    Q: torch.Tensor      # Q values
):
    assert len(F) == len(G) == len(Q), "All parameter arrays must have the same length"
    num_bands = len(F)
    
    # First band: Low Shelf
    response = RBJ_HighShelf(x, F[0], G[0], Q[0]).response

    # Middle bands: Bell filters
    for i in range(1, num_bands - 1):
        response *= RBJ_Bell(x, F[i], G[i], Q[i]).response

    # Last band: High Shelf
    response *= RBJ_HighShelf(x, F[-1], G[-1], Q[-1]).response

    return response

###############################################################################
# MAIN
###############################################################################
# set the range of frequencies that we evaluate over
f = torch.linspace(20, 20000, 2000)

num_bands = 5
Frequencies = np.array([])
GdB = np.array([])
Qs = np.array([])


for i in range(num_bands):
  Frequencies = np.append(Frequencies, frequency_denormalize(np.random.uniform()))
  GdB = np.append(GdB, gain_normalize(np.random.uniform()))
  Qs = np.append(Qs, q_denormalize(np.random.uniform()))

target_F = (torch.tensor(Frequencies))
target_G = (torch.tensor(10.0**(GdB/40)))
target_Q = (torch.tensor(Qs))

target_response = evaluate_mag_response(f, target_F, target_G, target_Q)

n_iters = 30000

parameters = torch.nn.ParameterList()

for i in range(num_bands):
  parameters.append(torch.tensor([np.random.rand(), np.random.rand(), np.random.rand()]))

optimizer = torch.optim.Adam(parameters, lr=0.06)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iters)


pbar = tqdm(range(n_iters))
for n in pbar:
  # reset gradients
  optimizer.zero_grad()

  # evaluate response of current solution
  pred_response = evaluate_mag_response(
      f,
      frequency_denormalize(torch.sigmoid(parameters[:][0])),
      gain_denormalize(torch.sigmoid(parameters[:][1])),
      q_denormalize(torch.sigmoid(parameters[:][2]))
    )



  # measure the MSE
  loss = torch.nn.functional.mse_loss((pred_response), (target_response))
  loss.backward()
  optimizer.step()
  scheduler.step()

  pbar.set_description(f"{loss.item():0.4e}")


import matplotlib.pyplot as plt

plt.semilogx(f, 20 * np.log10(pred_response.detach().numpy()), label="Prediction", linestyle='dotted', color='orange')

plt.plot(
    frequency_denormalize(torch.sigmoid(torch.tensor([band[0] for band in parameters]))).detach().numpy(),
    gain_normalize(torch.sigmoid(torch.tensor([band[1] for band in parameters]))).detach().numpy(),
    'o', color='orange'
)

plt.semilogx(f, 20 * np.log10(target_response.detach().numpy()), label="Target", linestyle='dotted', color='b')
plt.plot(target_F.detach().numpy(), 40*np.log10(target_G.detach().numpy()), 'o', color='b')

plt.legend()
plt.title("Frequency response matching using stochastic gradient descent")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.savefig("./match.png")
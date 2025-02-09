import torch
from tqdm import tqdm
import numpy as np
from Filters import *

###############################################################################
# UTILITIES
###############################################################################
def log_normalize(value, min, max):
  value = (torch.log(value) - torch.log(min)) / (torch.log(max) - torch.log((min)))
  return value

def log_denormalize(value, min, max):
  value = 10 ** (value * torch.log10(max) + (1.0 - value) * torch.log10(min))
  return value

def lin_normalize(value, min, max):
  value = ((value) * (max - min)) + min
  return value

def lin_denormalize(value, min, max):
  value = min + (max - min) * value
  return value


MIN_FREQ = 20.
MAX_FREQ = 20000.

def frequency_denormalize(f):
  min = torch.tensor(MIN_FREQ)
  max = torch.tensor(MAX_FREQ)
  return log_denormalize(f, min, max)

def frequency_normalize(f):
  min = torch.tensor(MIN_FREQ)
  max = torch.tensor(MAX_FREQ)
  return log_normalize(f, min, max)

MAX_GAIN_DB =  28.
MIN_GAIN_DB = - MAX_GAIN_DB

def gain_denormalize(g):
  min = torch.tensor(MIN_GAIN_DB)
  max = torch.tensor(MAX_GAIN_DB)
  return lin_denormalize(g, min, max)

def gain_normalize(g):
  min = torch.tensor(MIN_GAIN_DB)
  max = torch.tensor(MAX_GAIN_DB)
  return lin_normalize(g, min, max)

MIN_Q = 0.01
MAX_Q = 4.

def q_denormalize(q):
  min = torch.tensor(MIN_Q)
  max = torch.tensor(MAX_Q)
  return log_denormalize(q, min, max)

def q_normalize(q):
  min = torch.tensor(MIN_Q)
  max = torch.tensor(MAX_Q)
  return log_normalize(q, min, max)

def evaluate_mag_response(
    x: torch.Tensor,     # Frenquency vector
    F: torch.Tensor,     # Center frequencies
    G: torch.Tensor,     # Gain values
    Q: torch.Tensor      # Q values
):
    assert len(F) == len(G) == len(Q), "All parameter arrays must have the same length"
    NUM_OF_BANDS = len(F)
    
    # First band: Low Shelf
    response = RBJ_LowShelf(x, F[0], G[0], Q[0]).response

    # Middle bands: Bell filters
    for i in range(1, NUM_OF_BANDS - 1):
        response *= RBJ_Bell(x, F[i], G[i], Q[i]).response

    # Last band: High Shelf
    response *= RBJ_HighShelf(x, F[-1], G[-1], Q[-1]).response

    return response

###############################################################################
# LOAD FREQ RESPONSES
###############################################################################
NUM_OF_DELAYS = 6

import scipy.io 
target_matrix = scipy.io.loadmat("target_mag.mat")

f = torch.tensor(target_matrix.get('w')).squeeze()
target_response = torch.tensor(target_matrix.get('target_mag')).squeeze()

target_response = torch.pow(10.0, target_response / 20.0)

###############################################################################
# MAIN
###############################################################################
# set the range of frequencies that we evaluate over
NUM_OF_BANDS = 4
NUM_OF_ITER = 6000
# NUM_OF_FREQ = 512


SAMPLE_RATE = 48000

# f = torch.logspace(torch.log10(torch.tensor(20.)), torch.log10(torch.tensor(20000)), NUM_OF_FREQ)



# Frequencies = np.array([])
# GdB = np.array([])
# Qs = np.array([])


# for i in range(NUM_OF_BANDS):
#   Frequencies = np.append(Frequencies, frequency_denormalize(np.random.uniform()))
#   GdB = np.append(GdB, gain_normalize(np.random.uniform()))
#   Qs = np.append(Qs, q_denormalize(np.random.uniform()))

# target_F = (torch.tensor(Frequencies))
# target_G = (torch.tensor(GdB))
# target_Q = (torch.tensor(Qs))

# target_response = evaluate_mag_response(f, target_F, target_G, target_Q)


# rt60s = np.linspace(1., 0.1, NUM_OF_FREQ)
# rt60s[NUM_OF_FREQ // 2] = 0.3
# target_response = torch.pow(
#   torch.tensor(10.0, dtype=torch.float32),
#   torch.tensor((-60.0 / (SAMPLE_RATE * rt60s)) * (1500 / 20.0), dtype=torch.float32)
# )
# print(target_response)

parameters = torch.nn.ParameterList()

for i in range(NUM_OF_BANDS):
  parameters.append(torch.tensor([np.random.rand(), np.random.rand(), np.random.rand()]))

optimizer = torch.optim.Adam(parameters, lr=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_OF_ITER)


pbar = tqdm(range(NUM_OF_ITER))
for n in pbar:
  # reset gradients
  optimizer.zero_grad()
  # Extract parameters correctly as tensors
  freq_params = torch.stack([band[0] for band in parameters])
  gain_params = torch.stack([band[1] for band in parameters])
  q_params    = torch.stack([band[2] for band in parameters])
  # Apply sigmoid and denormalization
  pred_response = evaluate_mag_response(
      f,
      frequency_denormalize(torch.sigmoid(freq_params)),
      gain_denormalize(torch.sigmoid(gain_params)),
      q_denormalize(torch.sigmoid(q_params))
  )



  # measure the MSE
  loss = torch.nn.functional.mse_loss((pred_response), (target_response))
  loss.backward()
  # Scale frequency gradients to have more impact
  with torch.no_grad():
    parameters[0].grad *= 1  # Boost frequency gradients
    parameters[1].grad *= 1  # Moderate Q gradients
    parameters[2].grad *= 1  # Normal gain gradients

  optimizer.step()
  scheduler.step()

  pbar.set_description(f"{loss.item():0.4e}")


import matplotlib.pyplot as plt

plt.semilogx(f, 20 * np.log10(pred_response.detach().numpy()), label="Prediction", linestyle='dotted', color='orange')

low_shelf_response = RBJ_LowShelf(f, frequency_denormalize(torch.sigmoid(freq_params[0])), gain_denormalize(torch.sigmoid(gain_params[0])), q_denormalize(torch.sigmoid(q_params[0]))).response
high_shelf_response = RBJ_HighShelf(f, frequency_denormalize(torch.sigmoid(freq_params[-1])), gain_denormalize(torch.sigmoid(gain_params[-1])), q_denormalize(torch.sigmoid(q_params[-1]))).response

plt.semilogx(f, 20 * np.log10(low_shelf_response.detach().numpy()), label="LowShelf", linestyle='dotted', color='grey')
plt.semilogx(f, 20 * np.log10(high_shelf_response.detach().numpy()), label="HighShelf", color='grey')


plt.plot(
    frequency_denormalize(torch.sigmoid(torch.tensor([band[0] for band in parameters]))).detach().numpy(),
    gain_denormalize(torch.sigmoid(torch.tensor([band[1] for band in parameters]))).detach().numpy(),
    'o', color='orange'
)


plt.semilogx(f, 20 * np.log10(target_response.detach().numpy()), label="Target", linestyle='dotted', color='b')
# plt.plot(target_F.detach().numpy(), (target_G.detach().numpy()), 'o', color='b')



plt.legend()
plt.title("Frequency response matching using stochastic gradient descent")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.savefig("./match.png")
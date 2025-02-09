import torch
from tqdm import tqdm
import numpy as np
from Filters import *
import matplotlib.pyplot as plt
from utilities import *

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
SAMPLE_RATE = 48000

DELAYS = torch.randint(100, 3000, (NUM_OF_DELAYS, 1))
DELAYS, _ = torch.sort(DELAYS, dim=0)

import scipy.io 
target_matrix = scipy.io.loadmat("target_mag.mat")

f = torch.tensor(target_matrix.get('w')).squeeze()
target_response = torch.tensor(target_matrix.get('target_mag'),dtype=torch.float32).squeeze()

# target_response = torch.pow(10.0, target_response / 20.0)

# this is a theoretical array of response, we assume the first response was with a delay line of 1333 samples giving a max of 2s of RT60
DELAY_OF_EXAMPLE = 1333

RT_EXAMPLE = (-60 * DELAY_OF_EXAMPLE) / (SAMPLE_RATE * target_response)

target_responses = (-60 * DELAYS) / (SAMPLE_RATE * RT_EXAMPLE)
target_responses = torch.pow(10.0, target_responses / 20.0)

for i, delay in enumerate(DELAYS):
  plt.semilogx(f, 20 * np.log10(target_responses[i].T), label=f"{delay.item()} samples", linestyle='dotted')
plt.legend()
plt.title("Target Frequency Response of each delay line")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.savefig("./target.png")


###############################################################################
# MAIN
###############################################################################
# set the range of frequencies that we evaluate over
NUM_OF_BANDS = 4
NUM_OF_ITER = 100

parameters = torch.nn.ParameterList()

for i in range(NUM_OF_BANDS*NUM_OF_DELAYS):
  parameters.append(torch.nn.Parameter(torch.tensor([np.random.rand(), np.random.rand(), np.random.rand()], requires_grad=True)))

optimizer = torch.optim.Adam(parameters, lr=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_OF_ITER)
pred_responses = torch.zeros(NUM_OF_DELAYS, len(f), requires_grad=False).clone()

pbar = tqdm(range(NUM_OF_ITER))
for n in pbar:
  # reset gradients
  # optimizer.zero_grad()
  # Extract parameters correctly as tensors
  freq_params = torch.stack([band[0] for band in parameters])
  gain_params = torch.stack([band[1] for band in parameters])
  q_params    = torch.stack([band[2] for band in parameters])
  # Apply sigmoid and denormalization
  for i in range(NUM_OF_DELAYS):
    start_idx = i*NUM_OF_BANDS
    stop_idx = (i+1)*(NUM_OF_BANDS)
    # print(start_idx, stop_idx)
    pred_responses[i] = evaluate_mag_response(
        f,
        frequency_denormalize(torch.sigmoid(freq_params[start_idx:stop_idx])),
        gain_denormalize(torch.sigmoid(gain_params[start_idx:stop_idx])),
        q_denormalize(torch.sigmoid(q_params[start_idx:stop_idx]))
    )



  # measure the MSE
  loss = torch.nn.functional.mse_loss(pred_responses, target_responses)
  loss.backward(retain_graph=True)
  # Scale frequency gradients to have more impact
  # with torch.no_grad():
  #   parameters[0].grad *= 1  # Boost frequency gradients
  #   parameters[1].grad *= 1  # Moderate Q gradients
  #   parameters[2].grad *= 1  # Normal gain gradients

  optimizer.step()
  scheduler.step()

  pbar.set_description(f"{loss.item():0.4e}")




plt.semilogx(f, 20 * np.log10(pred_responses.detach().numpy().T), label="Prediction", linestyle='dotted', color='orange')

# low_shelf_response = RBJ_LowShelf(f, frequency_denormalize(torch.sigmoid(freq_params[0])), gain_denormalize(torch.sigmoid(gain_params[0])), q_denormalize(torch.sigmoid(q_params[0]))).response
# high_shelf_response = RBJ_HighShelf(f, frequency_denormalize(torch.sigmoid(freq_params[-1])), gain_denormalize(torch.sigmoid(gain_params[-1])), q_denormalize(torch.sigmoid(q_params[-1]))).response

# plt.semilogx(f, 20 * np.log10(low_shelf_response.detach().numpy()), label="LowShelf", linestyle='dotted', color='grey')
# plt.semilogx(f, 20 * np.log10(high_shelf_response.detach().numpy()), label="HighShelf", color='grey')


# plt.plot(
#     frequency_denormalize(torch.sigmoid(torch.tensor([band[0] for band in parameters]))).detach().numpy(),
#     gain_denormalize(torch.sigmoid(torch.tensor([band[1] for band in parameters]))).detach().numpy(),
#     'o', color='orange'
# )


# plt.semilogx(f, 20 * np.log10(target_response.detach().numpy()), label="Target", linestyle='dotted', color='b')
# plt.plot(target_F.detach().numpy(), (target_G.detach().numpy()), 'o', color='b')



plt.legend()
plt.title("Frequency response matching using stochastic gradient descent")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.savefig("./match.png")
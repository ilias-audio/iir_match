import torch
from tqdm import tqdm
import numpy as np
from Filters import *
import matplotlib.pyplot as plt
from utilities import *
import scipy.io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def evaluate_mag_response(
  x: torch.Tensor,     # Frequency vector
  F: torch.Tensor,     # Center frequencies (NUM_OF_DELAYS x NUM_OF_BANDS)
  G: torch.Tensor,     # Gain values (NUM_OF_DELAYS x NUM_OF_BANDS)
  Q: torch.Tensor      # Q values (NUM_OF_DELAYS x NUM_OF_BANDS)
):
  assert F.shape == G.shape == Q.shape, "All parameter arrays must have the same shape"
  NUM_OF_DELAYS, NUM_OF_BANDS = F.shape
  
  # Initialize response tensor
  response = torch.ones((NUM_OF_DELAYS, len(x)), device=x.device)
  
  # Compute responses for each band
  low_shelf_responses = RBJ_LowShelf(x, F[:, 0], G[:, 0], Q[:, 0]).response
  bell_responses = torch.stack([RBJ_Bell(x, F[:, i], G[:, i], Q[:, i]).response for i in range(1, NUM_OF_BANDS - 1)], dim=1)
  high_shelf_responses = RBJ_HighShelf(x, F[:, -1], G[:, -1], Q[:, -1]).response
  
  # Combine responses
  response *= low_shelf_responses.T
  response *= torch.prod(bell_responses, dim=1).T
  response *= high_shelf_responses.T

  return response

###############################################################################
# LOAD FREQ RESPONSES
###############################################################################
NUM_OF_DELAYS = 6
SAMPLE_RATE = 48000

DELAYS = torch.randint(100, 2000, (NUM_OF_DELAYS, 1), device=device)
DELAYS, _ = torch.sort(DELAYS, dim=0)

target_matrix = scipy.io.loadmat("target_mag.mat")

f = torch.tensor(target_matrix.get('w'), dtype=torch.float32).squeeze().to(device)
target_response = torch.tensor(target_matrix.get('target_mag'), dtype=torch.float32).squeeze().to(device)

DELAY_OF_EXAMPLE = 1500
RT_EXAMPLE = (-60 * DELAY_OF_EXAMPLE) / (SAMPLE_RATE * target_response)

target_responses = (-60 * DELAYS) / (SAMPLE_RATE * RT_EXAMPLE)
target_responses = torch.pow(10.0, target_responses / 20.0)

###############################################################################
# MAIN
###############################################################################
NUM_OF_BANDS = 8
NUM_OF_ITER = 10000

parameters = torch.nn.ParameterList([
  torch.nn.Parameter(torch.rand((3), requires_grad=True, device=device, dtype=torch.float32))
  for _ in range(NUM_OF_DELAYS*NUM_OF_BANDS)
])

optimizer = torch.optim.Adam(parameters, lr=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_OF_ITER)
pbar = tqdm(range(NUM_OF_ITER))

for n in pbar:  
  freq_params = torch.stack([param[0] for param in parameters]).view(NUM_OF_DELAYS, NUM_OF_BANDS)
  gain_params = torch.stack([param[1] for param in parameters]).view(NUM_OF_DELAYS, NUM_OF_BANDS)
  q_params    = torch.stack([param[2] for param in parameters]).view(NUM_OF_DELAYS, NUM_OF_BANDS)

  f_expanded = f.unsqueeze(0).repeat(NUM_OF_DELAYS, 1)
  f_expanded = f_expanded.T

  pred_responses = evaluate_mag_response(
    f_expanded,
    frequency_denormalize(torch.sigmoid(freq_params)),
    gain_denormalize(torch.sigmoid(gain_params)),
    q_denormalize(torch.sigmoid(q_params))
  )

  if n % 1000 == 0:
    # print("Freqs: ", frequency_denormalize(torch.sigmoid(freq_params)).tolist())
    # print("Gain: " , gain_denormalize(torch.sigmoid(gain_params)).tolist())
    # print("Q: "    , q_denormalize(torch.sigmoid(q_params)).tolist())
    # print("Freqs: ", freq_params.tolist())
    # print("Gain: " , gain_params.tolist())
    # print("Q: "    , q_params.tolist())

    
    plt.clf()
    plt.semilogx(f.cpu(), 20 * np.log10(pred_responses.detach().cpu().numpy().T), label="Prediction")
    plt.semilogx(f.cpu(), 20 * np.log10(target_responses.detach().cpu().numpy().T), label="Target", linestyle='dotted')
    plt.plot(frequency_denormalize(torch.sigmoid(freq_params)).detach().cpu(), gain_denormalize(torch.sigmoid(gain_params)).detach().cpu().numpy(), 'o')
    plt.legend()
    plt.title("Frequency response matching using stochastic gradient descent")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.savefig("./figures/match"+str(n)+".png")

  loss = torch.zeros(NUM_OF_DELAYS, device=device)
  optimizer.zero_grad()
  for i in range(NUM_OF_DELAYS):
    loss[i] = torch.nn.functional.mse_loss(20*torch.log10(pred_responses[i]), 20*torch.log10(target_responses[i]))
    loss[i].backward(retain_graph=True)
  optimizer.step()
  scheduler.step()

  pbar.set_description(f"{loss[0].item():0.4e}")


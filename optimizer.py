import torch
from tqdm import tqdm
import numpy as np
from Filters import *
import matplotlib.pyplot as plt
from utilities import *
import scipy.io
import scipy.signal as signal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") # still too slow on GPU
SAMPLE_RATE = 48000
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

def sos_mag_response(F: torch.Tensor,     # Center frequencies (NUM_OF_DELAYS x NUM_OF_BANDS)
  G: torch.Tensor,     # Gain values (NUM_OF_DELAYS x NUM_OF_BANDS)
  Q: torch.Tensor,      # Q values (NUM_OF_DELAYS x NUM_OF_BANDS)
  Fs: torch.Tensor     # Sample rate
  ):
    assert F.shape == G.shape == Q.shape, "All parameter arrays must have the same shape"
    NUM_OF_DELAYS, NUM_OF_BANDS = F.shape
    
    for i in range(NUM_OF_BANDS):
      if i == 0:
        low_shelf = RBJ_LowShelf(Fs, F[:, 0], G[:, 0], Q[:, 0], sample_rate=Fs)
        low_shelf.compute_sos()
        sos = low_shelf.sos
        # print(sos.shape)
      elif i == NUM_OF_BANDS - 1:
        high_shelf = RBJ_HighShelf(Fs, F[:, -1], G[:, -1], Q[:, -1], sample_rate=Fs)
        high_shelf.compute_sos()
        sos = torch.cat((sos, high_shelf.sos), dim=0)
      else:
        bell = RBJ_Bell(Fs, F[:, i], G[:, i], Q[:, i], sample_rate=Fs)
        bell.compute_sos()
        sos = torch.cat((sos, bell.sos), dim=0)
    # print(sos.shape)
    return signal.sosfreqz(sos.detach().numpy(), worN=2028, fs=Fs)
    


def convert_proto_gain_to_delay(gamma, delays, fs):
  gain = gamma * (delays / fs)
  return gain
  


###############################################################################
# LOAD FREQ RESPONSES
###############################################################################
NUM_OF_DELAYS = 3


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
NUM_OF_BANDS = 4
NUM_OF_ITER = 10001

parameters = torch.nn.ParameterList()

parameters.append(torch.rand(NUM_OF_BANDS,requires_grad=True, device=device, dtype=torch.float32)) # Common Freqs
parameters.append(torch.rand(NUM_OF_BANDS, requires_grad=True, device=device, dtype=torch.float32)) # Common gains
parameters.append(torch.rand(NUM_OF_BANDS, requires_grad=True, device=device, dtype=torch.float32)) # Common Q



optimizer = torch.optim.Adam(parameters, lr=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_OF_ITER)
pbar = tqdm(range(NUM_OF_ITER))

for n in pbar:  

  freq_params = torch.sigmoid(parameters[0].unsqueeze(0).repeat(NUM_OF_DELAYS, 1))
  gain_params = torch.sigmoid(parameters[1])
  q_params = torch.sigmoid(parameters[2].unsqueeze(0).repeat(NUM_OF_DELAYS, 1))

  f_expanded = f.unsqueeze(0).repeat(NUM_OF_DELAYS, 1)
  f_expanded = f_expanded.T

  pred_responses = evaluate_mag_response(
    f_expanded,
    frequency_denormalize((freq_params)),
    convert_proto_gain_to_delay(gain_denormalize((gain_params)), DELAYS, SAMPLE_RATE),
    q_denormalize((q_params))
  )
  
  sos_freq, sos_mag = sos_mag_response(
    frequency_denormalize((freq_params)),
    convert_proto_gain_to_delay(gain_denormalize((gain_params)), DELAYS, SAMPLE_RATE),
    q_denormalize((q_params)),
    SAMPLE_RATE
  )

  if n % 1000 == 0:
        
    plt.clf()
    plt.semilogx(f.cpu(), 20 * np.log10(pred_responses.detach().cpu().numpy().T), label="Prediction")
    plt.semilogx(f.cpu(), 20 * np.log10(target_responses.detach().cpu().numpy().T), label="Target", linestyle='dotted')
    plt.legend()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.savefig("./figures/match"+str(n)+".png")

  loss = torch.zeros(1, device=device)
  optimizer.zero_grad()
  loss = torch.nn.functional.mse_loss(20 * torch.log10((pred_responses)), 20 * torch.log10(target_responses))
  loss.backward(retain_graph=False)
  optimizer.step()
  scheduler.step()

  pbar.set_description(f"{loss:0.4e}")


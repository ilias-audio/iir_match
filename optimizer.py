import torch
from tqdm import tqdm
import numpy as np
from Filters import *
import matplotlib.pyplot as plt
from utilities import *
import scipy.io
import scipy.signal as signal
import os 



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


def indivual_mag_response( x: torch.Tensor,     # Frequency vector
  F: torch.Tensor,     # Center frequencies (NUM_OF_DELAYS x NUM_OF_BANDS)
  G: torch.Tensor,     # Gain values (NUM_OF_DELAYS x NUM_OF_BANDS)
  Q: torch.Tensor      # Q values (NUM_OF_DELAYS x NUM_OF_BANDS)
):
  assert F.shape == G.shape == Q.shape, "All parameter arrays must have the same shape"
  NUM_OF_DELAYS, NUM_OF_BANDS = F.shape
  
  # Initialize response tensor
  response = torch.ones((NUM_OF_BANDS, len(x)), device=x.device)
  
  # Compute responses for each band
  low_shelf_responses = RBJ_LowShelf(x, F[:, 0], G[:, 0], Q[:, 0]).response
  bell_responses = torch.stack([RBJ_Bell(x, F[:, i], G[:, i], Q[:, i]).response for i in range(1, NUM_OF_BANDS - 1)], dim=1)
  high_shelf_responses = RBJ_HighShelf(x, F[:, -1], G[:, -1], Q[:, -1]).response
  
  # Combine responses
  response[0,:] *= low_shelf_responses.T.squeeze()
  for i in range(1, NUM_OF_BANDS - 1):
    response[i,:] *= bell_responses[:,i-1].T.squeeze()
  response[-1, :] *= high_shelf_responses.T.squeeze()

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
      elif i == NUM_OF_BANDS - 1:
        high_shelf = RBJ_HighShelf(Fs, F[:, -1], G[:, -1], Q[:, -1], sample_rate=Fs)
        high_shelf.compute_sos()
        sos = torch.cat((sos, high_shelf.sos), dim=0)
      else:
        bell = RBJ_Bell(Fs, F[:, i], G[:, i], Q[:, i], sample_rate=Fs)
        bell.compute_sos()
        sos = torch.cat((sos, bell.sos), dim=0)
    return signal.sosfreqz(sos.detach().cpu().numpy(), worN=2028, fs=Fs)
    


def convert_proto_gain_to_delay(gamma, delays, fs):
  gain = gamma * (delays / fs)
  return gain
  


###############################################################################
# LOAD FREQ RESPONSES those are utils
###############################################################################
def load_target_mag(device):
  target_matrix = scipy.io.loadmat("target_mag.mat")
  f = torch.tensor(target_matrix.get('w'), dtype=torch.float32).squeeze().to(device)
  target_response = torch.tensor(target_matrix.get('target_mag'), dtype=torch.float32).squeeze().to(device)

  DELAY_OF_EXAMPLE = 1500
  RT_EXAMPLE = (-60 * DELAY_OF_EXAMPLE) / (SAMPLE_RATE * target_response)

  target_responses = (-60 * DELAYS) / (SAMPLE_RATE * RT_EXAMPLE)
  target_responses = torch.pow(10.0, target_responses / 20.0)
  return f, target_responses


def load_dataset_mag(index,`` device):
  rt_dataset = np.load(os.path.join("data", "interpolated_dataset.npy")) # should have the interpolation shape in the name
  target_responses = (-60 * DELAYS.cpu()) / (SAMPLE_RATE * rt_dataset[:, index])
  target_responses = torch.pow(10.0, target_responses / 20.0)
  f = np.logspace(np.log10(20), np.log10(20000), rt_dataset.shape[0])
  return torch.tensor(f, dtype=torch.float32, device=device), torch.tensor(target_responses, dtype=torch.float32, device=device)
  


def response_to_rt(response, delay):
  rt = (-60 * delay) / (SAMPLE_RATE * (20* torch.log10(response)))
  return rt

###############################################################################
# MAIN
###############################################################################

class MatchEQ:
  def __init__(self, device , num_of_iter, num_of_bands, num_of_delays, sample_rate):
    self.sample_rate = sample_rate
    self.device = torch.device(device)
    self.num_of_bands = num_of_bands
    self.num_of_delays = num_of_delays
    self.min_delay_in_samples = int()
    self.max_delay_in_samples = 15000
    self.num_of_iter = num_of_iter
    self.delays = torch.sort(torch.randint(self.min_delay_in_samples, self.max_delay_in_samples, (self.num_of_delays, 1), device=self.device), dim=0)





if __name__ == "__main__":

  device = "cuda" if torch.cuda.is_available() else "cpu"

  SAMPLE_RATE     = 48000
  NUM_OF_DELAYS   = 1
  NUM_OF_BANDS    = 6
  NUM_OF_ITER     = 301

  MatchEQ = MatchEQ(device, NUM_OF_ITER, NUM_OF_BANDS, NUM_OF_DELAYS, SAMPLE_RATE)
  Dataset = Dataloader(os.join("data","interpolated_dataset.npy"), device)
  
  trained_gains = torch.zeros((SIZE_OF_DATASET, NUM_OF_BANDS), device=device)
  trained_frequencies = torch.zeros((SIZE_OF_DATASET, NUM_OF_BANDS), device=device)
  trained_q = torch.zeros((SIZE_OF_DATASET, NUM_OF_BANDS), device=device)
  training_error = torch.zeros((SIZE_OF_DATASET, len(rt_dataset[0])), device=device)
  
  
  import concurrent.futures

  def process_rt_index(rt_index):
    f, target_responses = load_dataset_mag(rt_index, device)

  

    trained_rt = torch.zeros((len(f)), device=device)
    training_error = torch.zeros((len(f)), device=device)

    # optimizer = torch.optim.Adam(parameters, lr=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_OF_ITER)
    pbar = tqdm(range(NUM_OF_ITER), desc=f"RT Index {rt_index}")

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

      if n == NUM_OF_ITER - 1:
        individual_responses = indivual_mag_response(
          f_expanded,
          frequency_denormalize((freq_params)),
          convert_proto_gain_to_delay(gain_denormalize((gain_params)), DELAYS, SAMPLE_RATE),
          q_denormalize((q_params))
        )

        fig, ax = plt.subplots()
        ax.semilogx(f.cpu(), 20 * np.log10(pred_responses.detach().cpu().numpy().T), label="Prediction")
        ax.semilogx(f.cpu(), 20 * np.log10(target_responses.detach().cpu().numpy().T), label="Target", linestyle='dotted')
        ax.semilogx(f.cpu(), 20 * np.log10(individual_responses.detach().cpu().numpy().T), label="individual", linestyle='dotted', color='grey')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        fig.savefig(f"./figures/match_rt{rt_index}.png")
        plt.close(fig)

      optimizer.zero_grad()
      loss = torch.nn.functional.mse_loss(20 * torch.log10((pred_responses[-1, :])), 20 * torch.log10(target_responses[-1, :]))
      loss.backward(retain_graph=False)
      optimizer.step()
      scheduler.step()

    trained_gains[rt_index, :] = convert_proto_gain_to_delay(gain_denormalize((gain_params)), DELAYS, SAMPLE_RATE)
    trained_frequencies[rt_index, :] = frequency_denormalize((freq_params))
    trained_q[rt_index, :] = q_denormalize((q_params))
    trained_rt[:] = response_to_rt(pred_responses[-1, :], DELAYS)
    training_error[:] = ((rt_dataset[rt_index, :] - trained_rt[:]) / rt_dataset[rt_index, :]) * 100

    return rt_index, trained_gains[rt_index, :], trained_frequencies[rt_index, :], trained_q[rt_index, :], training_error[:]


  with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_rt_index, range(1000)))

  for rt_index, gains, frequencies, q_values, errors in results:
    trained_gains[rt_index, :] = gains
    trained_frequencies[rt_index, :] = frequencies
    trained_q[rt_index, :] = q_values
    training_error[rt_index, :] = errors

  np.save("trained_gains.npy", trained_gains.detach().cpu().numpy())
  np.save("trained_frequencies.npy", trained_frequencies.detach().cpu().numpy())
  np.save("trained_q.npy", trained_q.detach().cpu().numpy())
  np.save("training_error.npy", training_error.detach().cpu().numpy())


  # load error and compute probability
  training_error = np.load("training_error.npy")
  plt.clf()
  plt.hist(training_error.flatten(), density=True, histtype='step', log=True, bins=SIZE_OF_DATASET // 100)
  plt.xlabel("Percentage Error")
  plt.ylabel("Probability")
  plt.title("T_{60} Error Distribution")
  plt.savefig(os.path.join("figures", "figure_4.png"))

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
def load_target_mag():
  target_matrix = scipy.io.loadmat("target_mag.mat")
  f = torch.tensor(target_matrix.get('w'), dtype=torch.float32).squeeze().to(device)
  target_response = torch.tensor(target_matrix.get('target_mag'), dtype=torch.float32).squeeze().to(device)

  DELAY_OF_EXAMPLE = 1500
  RT_EXAMPLE = (-60 * DELAY_OF_EXAMPLE) / (SAMPLE_RATE * target_response)

  target_responses = (-60 * DELAYS) / (SAMPLE_RATE * RT_EXAMPLE)
  target_responses = torch.pow(10.0, target_responses / 20.0)
  return f, target_responses


def load_dataset_mag(index):
  rt_dataset = np.load("interpolated_dataset.npy")
  target_responses = (-60 * DELAYS) / (SAMPLE_RATE * rt_dataset[:, index])
  target_responses = torch.pow(10.0, target_responses / 20.0)
  f = np.logspace(np.log10(20), np.log10(20000), rt_dataset.shape[0])
  return torch.tensor(f, dtype=torch.float32), torch.tensor(target_responses, dtype=torch.float32)
  


def response_to_rt(response, delay):
  rt = (-60 * delay) / (SAMPLE_RATE * (20* torch.log10(response)))
  # print(rt)
  return rt

###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":


  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  device = torch.device("cpu") # still too slow on GPU
  SAMPLE_RATE = 48000
  NUM_OF_DELAYS = 1
  DELAYS = torch.randint(200, 15000, (NUM_OF_DELAYS, 1), device=device)
  DELAYS, _ = torch.sort(DELAYS, dim=0)

  NUM_OF_BANDS = 6
  NUM_OF_ITER = 3001
  SIZE_OF_DATASET = 1000
  trained_gains = torch.zeros((SIZE_OF_DATASET, NUM_OF_BANDS))
  trained_frequencies = torch.zeros((SIZE_OF_DATASET, NUM_OF_BANDS))
  trained_q = torch.zeros((SIZE_OF_DATASET, NUM_OF_BANDS))

  rt_dataset = np.load("interpolated_dataset.npy")
  rt_dataset = torch.tensor(rt_dataset.T)



  for rt_index in range(1000):
    f, target_responses = load_dataset_mag(rt_index)

    parameters = torch.nn.ParameterList()

    freq_values = np.ones(NUM_OF_BANDS)
    freq_values[0:-2] = np.logspace(np.log10(20), np.log10(16000), NUM_OF_BANDS-2)
    freq_values[0] = 1000
    freq_values[-1] = 1000
    
    # print(freq_values)
    parameters.append(frequency_normalize(torch.tensor(
      freq_values,
      requires_grad=True, device=device, dtype=torch.float32
    ))) # Common Freqs
    parameters.append(torch.zeros(NUM_OF_BANDS, requires_grad=True, device=device, dtype=torch.float32)) # Common gains
    parameters.append(torch.zeros(NUM_OF_BANDS, requires_grad=True, device=device, dtype=torch.float32)) # Common Q


    trained_rt = torch.zeros((SIZE_OF_DATASET,len(f)))
    training_error = torch.zeros((SIZE_OF_DATASET, len(f)))

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

      if n == NUM_OF_ITER-1:
        individual_responses = indivual_mag_response(
        f_expanded,
        frequency_denormalize((freq_params)),
        convert_proto_gain_to_delay(gain_denormalize((gain_params)), DELAYS, SAMPLE_RATE),
        q_denormalize((q_params))
      )

            
        plt.clf()
        plt.semilogx(f.cpu(), 20 * np.log10(pred_responses.detach().cpu().numpy().T), label="Prediction")
        plt.semilogx(f.cpu(), 20 * np.log10(target_responses.detach().cpu().numpy().T), label="Target", linestyle='dotted')
        plt.semilogx(f.cpu(), 20 * np.log10(individual_responses.detach().cpu().numpy().T), label="individual", linestyle='dotted', color= 'grey')
        # plt.legend()
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.savefig("./figures/match_rt"+str(rt_index)+".png")

      # loss = torch.zeros(1, device=device)
      optimizer.zero_grad()
      loss = torch.nn.functional.mse_loss(20 * torch.log10((pred_responses[-1,:])), 20 * torch.log10(target_responses[-1,:]))
      loss.backward(retain_graph=False)
      optimizer.step()
      scheduler.step()

      pbar.set_description(f"{loss:0.4e}")

    ## save trained parameters
    trained_gains[rt_index, :] = convert_proto_gain_to_delay(gain_denormalize((gain_params)), DELAYS, SAMPLE_RATE)
    trained_frequencies[rt_index, :] = frequency_denormalize((freq_params))
    trained_q[rt_index, :]= q_denormalize((q_params))
    trained_rt[rt_index, :] = response_to_rt(pred_responses[-1, :], DELAYS)
    training_error[rt_index, :] = ((rt_dataset[rt_index,:] - trained_rt[rt_index, :]) / rt_dataset[rt_index,:]) * 100
    # print("Error: ", training_error[rt_index, :])
    # print("Error: ", max(training_error[rt_index, :]))
    # print("Error: ", min(training_error[rt_index, :]))

  np.save("trained_gains.npy", trained_gains.detach().cpu().numpy())
  np.save("trained_frequencies.npy", trained_frequencies.detach().cpu().numpy())
  np.save("trained_q.npy", trained_q.detach().cpu().numpy())
  np.save("training_error.npy", training_error.detach().cpu().numpy())


  # load error and compute probability
  training_error = np.load("training_error.npy")
  plt.clf()
  plt.hist(training_error.flatten(), density=True, histtype='step', log=True, bins=SIZE_OF_DATASET // 100)
  # plt.xlim(-100, 100)
  plt.xlabel("Percentage Error")
  plt.ylabel("Probability")
  plt.title("T_{60} Error Distribution")
  plt.savefig(os.path.join("figures", "figure_4.png"))
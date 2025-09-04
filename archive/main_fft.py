import torch
import torchaudio
import Filters
from scipy import signal
import MatchFFT
if __name__ == "__main__":
  ######### TOP LEVEL PARAMETERS ###########
  SAMPLE_RATE = 48000
  NUM_OF_DELAYS = 1
  NUM_OF_BANDS = 6
  NUM_OF_ITER = 1000
  FFT_SIZE = 2**12
  ##########################################


'''
Step 1) DONE Create WGN 
Step 2) DONE Create an FFT 
Step 3) DONE Add Auraloss MS-STFT
Step 4) DONE Add target filters 
Step 5) DONE Add candidate filter
Step 6) DONE Backpropagate

'''
device = torch.device("cpu")

EQ_SIZE = (NUM_OF_BANDS, NUM_OF_DELAYS)

F = torch.randint(20, 16000, EQ_SIZE, device=device, dtype=torch.float32)  # Random frequencies in Hz
G = (torch.rand(EQ_SIZE, device=device, dtype=torch.float32) - 0.5) * 6
Q = (torch.rand(EQ_SIZE, device=device, dtype=torch.float32) + 0.1) * 5 

print(f"Target Frequencies: {F.squeeze()}")
print(f"Target Gains: {G.squeeze()}")
print(f"Target Q: {Q.squeeze()}")

x = torch.zeros((512,1), device=device, dtype=torch.float32)



## Create an EQ curve
# Get SOS arrays for each filter section
low_shelf_sos = Filters.RBJ_LowShelf(x, F[0, :], G[0, :], Q[0, :]).sos  # shape: (:,6)
bell_sos = torch.cat([Filters.RBJ_Bell(x, F[i, :], G[i, :], Q[i, :]).sos for i in range(1, NUM_OF_BANDS-1)], dim=0)  # shape: (:,6)
high_shelf_sos = Filters.RBJ_HighShelf(x, F[-1, :], G[-1, :], Q[-1, :]).sos  # shape: (:,6)

# Concatenate all sos sections along dim=0
all_sos = torch.cat([low_shelf_sos, bell_sos, high_shelf_sos], dim=0)  # shape: (num_sections, 6)

# Split into b_coeffs (first 3 columns) and a_coeffs (last 3 columns)
b_coeffs = all_sos[:, :3] # shape: (num_sections * 3,)
a_coeffs = all_sos[:, 3:]  # shape: (num_sections * 3,)


### Create a WGN Generator
input_signal = torch.randn((1, SAMPLE_RATE), device=device)  # 2 seconds
input_signal /= input_signal.std()  # Normalize to unit variance for flat spectrum

# Apply cascaded filtering using torchaudio.lfilter
target_filtered_signal = input_signal.clone()

for i in range(all_sos.shape[0]):
  # Extract b and a coefficients for this section
  b = all_sos[i, :3]  # numerator coefficients
  a = all_sos[i, 3:]  # denominator coefficients
  
  # Apply lfilter section by section using torchaudio
  target_filtered_signal = torchaudio.functional.lfilter(
    target_filtered_signal, 
    a, 
    b, 
    clamp=False
  )
## Create an STFT

signal_stft = torch.stft(target_filtered_signal, n_fft=FFT_SIZE, hop_length=FFT_SIZE//2, window=torch.hann_window(FFT_SIZE, device=device), return_complex=True)



MatchFFT = MatchFFT.MatchFFT(
    num_of_iter=NUM_OF_ITER,
    num_of_bands=NUM_OF_BANDS,
    num_of_delays=NUM_OF_DELAYS,
    sample_rate=SAMPLE_RATE,
    fft_size=FFT_SIZE,
    device=device
)

MatchFFT.train(target_filtered_signal, input_signal)




# Add validation after training
with torch.no_grad():
    from utilities import *
    # MatchFFT.parameters[0] = torch.special.logit(frequency_normalize(F).squeeze(), eps = 1e-6)
    # MatchFFT.parameters[1] = torch.special.logit(gain_normalize(G*10).squeeze(), eps = 1e-6)
    # MatchFFT.parameters[2] = torch.special.logit(q_normalize(Q).squeeze(), eps = 1e-6)
    final_prediction = MatchFFT.forward(input_signal)
    final_loss = MatchFFT.loss_function(final_prediction.unsqueeze(0).unsqueeze(0), target_filtered_signal.unsqueeze(0))
    print(f"Final validation loss: {final_loss.item():.6f}")
    
    # Compute frequency domain error
    target_fft = torch.fft.rfft(target_filtered_signal.squeeze())
    pred_fft = torch.fft.rfft(final_prediction.squeeze())
    freq_error = torch.mean(torch.abs(torch.abs(target_fft) - torch.abs(pred_fft)))
    print(f"Frequency domain error: {freq_error.item():.6f}")

pred_F = MatchFFT.eq_parameters_freqs
pred_G = MatchFFT.eq_parameters_gains
pred_Q = MatchFFT.eq_parameters_q



## Create an EQ curve
# Get SOS arrays for each filter section
low_shelf_sos = Filters.RBJ_LowShelf(x, pred_F[0, :], pred_G[0, :], pred_Q[0, :]).sos  # shape: (:,6)
bell_sos = torch.cat([Filters.RBJ_Bell(x, pred_F[i, :], pred_G[i, :], pred_Q[i, :]).sos for i in range(1, NUM_OF_BANDS-1)], dim=0)  # shape: (:,6)
high_shelf_sos = Filters.RBJ_HighShelf(x, pred_F[-1, :], pred_G[-1, :], pred_Q[-1, :]).sos  # shape: (:,6)

# Concatenate all sos sections along dim=0
all_sos = torch.cat([low_shelf_sos, bell_sos, high_shelf_sos], dim=0)  # shape: (num_sections, 6)

# Split into b_coeffs (first 3 columns) and a_coeffs (last 3 columns)
b_coeffs = all_sos[:, :3] # shape: (num_sections * 3,)
a_coeffs = all_sos[:, 3:]  # shape: (num_sections * 3,)


### Create a WGN Generator
input_signal = torch.randn((1,SAMPLE_RATE), device=device) ## 2 second of gaussian white noise
input_signal /= input_signal.std()  # Normalize to unit variance for flat spectrum

# Apply cascaded filtering using torchaudio.lfilter
pred_filtered_signal = input_signal.clone()

for i in range(all_sos.shape[0]):
  # Extract b and a coefficients for this section
  b = all_sos[i, :3]  # numerator coefficients
  a = all_sos[i, 3:]  # denominator coefficients
  
  # Apply lfilter section by section using torchaudio
  pred_filtered_signal = torchaudio.functional.lfilter(
    pred_filtered_signal, 
    a, 
    b, 
    clamp=False
  )


pred_stft = torch.stft(pred_filtered_signal, n_fft=FFT_SIZE, hop_length=FFT_SIZE//2, window=torch.hann_window(FFT_SIZE, device=device), return_complex=True)



import matplotlib.pyplot as plt
import numpy as np

# Plot STFT magnitude (dB)
plt.figure("stft_target")
# Calculate frequency bins
freqs = torch.fft.fftfreq(FFT_SIZE, 1/SAMPLE_RATE)[:FFT_SIZE//2+1]



plt.imshow(
  20 * torch.log10(signal_stft.abs().squeeze().cpu() + 1e-8).numpy(),
  aspect='auto',
  origin='lower',
  interpolation='nearest',
  extent=[0, signal_stft.shape[-1], freqs[0], freqs[-1]]
)
plt.title("STFT Magnitude (dB)")
plt.xlabel("Frame")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="dB")

plt.savefig('stft_spectrum_1.png')
# Compute and plot long-term spectrum (mean over time)
long_term_spectrum = signal_stft.abs().mean(dim=-1).squeeze().cpu().numpy() / FFT_SIZE
pred_long_term_spectrum = pred_stft.abs().mean(dim=-1).squeeze().cpu().detach().numpy() / FFT_SIZE

response = Filters.evaluate_mag_response(
  abs(freqs[:len(long_term_spectrum)]),     # Frequency vector
  MatchFFT.eq_parameters_freqs,     # Center frequencies (NUM_OF_DELAYS x NUM_OF_BANDS)
  MatchFFT.eq_parameters_gains,     # Gain values (NUM_OF_DELAYS x NUM_OF_BANDS)
  MatchFFT.eq_parameters_q      # Q values (NUM_OF_DELAYS x NUM_OF_BANDS)
)

target_param = Filters.evaluate_mag_response(
  abs(freqs[:len(long_term_spectrum)]),     # Frequency vector
  F,     # Center frequencies (NUM_OF_DELAYS x NUM_OF_BANDS)
  G,     # Gain values (NUM_OF_DELAYS x NUM_OF_BANDS)
  Q      # Q values (NUM_OF_DELAYS x NUM_OF_BANDS)
)

plt.figure("long_term_spectrum")
plt.semilogx(abs(freqs[:len(long_term_spectrum)]), 20 * np.log10(long_term_spectrum + 1e-8), color='blue')
plt.semilogx(abs(freqs[:len(long_term_spectrum)]), 20 * np.log10(pred_long_term_spectrum + 1e-8), color='red')
plt.semilogx(abs(freqs[:len(long_term_spectrum)]), 20 * np.log10(abs(target_param.detach().numpy()/32)), color='blue', linestyle='--')
plt.semilogx(abs(freqs[:len(long_term_spectrum)]), 20 * np.log10(abs(response.detach().numpy()/32)), color='red', linestyle='--')
plt.title("Long-term Spectrum (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")

plt.savefig('stft_spectrum_2.png')
import torch
import torchaudio
from Dataloader import Dataloader
from utilities import *
from Filters import evaluate_mag_response, evaluate_sos_response, RBJ_LowShelf, RBJ_Bell, RBJ_HighShelf
from tqdm import tqdm
from scipy import signal
import auraloss
import os
import PrintUtilities

class MatchFFT:
    def __init__(self, num_of_iter: int, num_of_bands: int, num_of_delays: int, sample_rate: int, fft_size: int, device: str):
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.num_of_bands = num_of_bands
        self.num_of_delays = num_of_delays
        self.fft_size = fft_size
        self.min_delay_in_seconds = 0.003
        self.max_delay_in_seconds = 0.1
        self.min_delay_in_samples = int(self.min_delay_in_seconds * self.sample_rate)
        self.max_delay_in_samples = int(self.max_delay_in_seconds * self.sample_rate)
        self.num_of_iter = num_of_iter
        # self.delays, _ = torch.sort(torch.randint(self.min_delay_in_samples, self.max_delay_in_samples, (self.num_of_delays, 1), device=self.device, dtype=torch.float32), dim=0)
        self.delays = torch.tensor(4800, device=self.device, dtype=torch.float32).repeat(self.num_of_delays, 1)
        assert self.delays.shape == (self.num_of_delays, 1), "Delays should be a column vector"

        self.loss_function = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[2**14, 2**15, 2**16],
            hop_sizes=[256, 512, 2048],
            win_lengths=[1024, 2048, 8192],
            n_bins=256,
            sample_rate=sample_rate,
        )

        self.init_training_parameters()
        self.setup_optimizer()

    def rfft_loss(self, prediction, target):
        target_fft = torch.fft.rfft(target.squeeze())
        pred_fft = torch.fft.rfft(prediction.squeeze())
        freq_error = torch.mean(torch.abs(torch.abs(target_fft) - torch.abs(pred_fft)))
        return freq_error

    def init_training_parameters(self):
        self.parameters = torch.nn.ParameterList()
        init_min_freq = torch.log10(torch.tensor(100.))
        init_max_freq = torch.log10(torch.tensor(16000.))

        freq_values = torch.ones(self.num_of_bands, requires_grad=False, device=self.device, dtype=torch.float32)
        freq_values[1:-1] = torch.logspace(init_min_freq, init_max_freq, self.num_of_bands-2)
        # Make the shelfs centered at 1000 Hz
        freq_values[0] = 1000 
        freq_values[-1] = 1000
        # freq_values.requires_grad(True)
        freq_values.requires_grad_(True)

        self.parameters.append(torch.special.logit((frequency_normalize(freq_values)), eps=1e-6))  # Common Freqs
        self.parameters.append(torch.special.logit(gain_normalize(torch.ones(self.num_of_bands, requires_grad=True, device=self.device, dtype=torch.float32) * -1.), eps=1e-6))  # Common gains
        self.parameters.append(torch.special.logit(q_normalize(torch.ones(self.num_of_bands, requires_grad=True, device=self.device, dtype=torch.float32) * 1.), eps=1e-6))  # Common Q
        
        assert self.parameters[0].shape == torch.Size([self.num_of_bands]), f"Frequency values should be a column vector, but got {self.parameters[0].shape}"
        assert self.parameters[1].shape == torch.Size([self.num_of_bands]), f"Gain values should be a column vector, but got {self.parameters[1].shape}"
        assert self.parameters[2].shape == torch.Size([self.num_of_bands]), f"Q values should be a column vector, but got {self.parameters[2].shape}"
    
    def setup_optimizer(self):
        # Use different learning rates for different parameters
        param_groups = [
            {'params': [self.parameters[0]], 'lr': 0.1},  # Frequencies
            {'params': [self.parameters[1]], 'lr': 0.1},   # Gains 
            {'params': [self.parameters[2]], 'lr': 0.1},   # Q values 
        ]
        self.optimizer = torch.optim.Adam(param_groups)
        # Use ReduceLROnPlateau for better convergence
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.num_of_iter)

    def train(self, target_x: torch.Tensor, input_signal: torch.Tensor):
        pbar = tqdm(range(self.num_of_iter), desc=f"Match FFT")
        best_loss = float('inf')
        patience_counter = 0

        for n in pbar:
            self.optimizer.zero_grad()

            prediction_x = self.forward(input_signal)
         
            target_x = target_x.unsqueeze(0)
            prediction_x = prediction_x.unsqueeze(0).unsqueeze(0)
            # print(target_x.shape)
            # print(prediction_x.shape)
            # stft_loss = self.loss_function(prediction_x, target_x)
            stft_loss = self.rfft_loss(prediction_x, target_x)
            
        
            # time_loss = torch.nn.functional.mse_loss(target_x, prediction_x)
            
        
            # target_mag = torch.abs(torch.stft(target_x, n_fft=1024, hop_length=256, window=torch.hann_window(1024, device=self.device), return_complex=True))
            # pred_mag = torch.abs(torch.stft(prediction_x, n_fft=1024, hop_length=256, window=torch.hann_window(1024, device=self.device), return_complex=True))
            # spectral_loss = torch.nn.functional.mse_loss(pred_mag.mean(dim=-1), target_mag.mean(dim=-1))
            
            # Combined loss
            loss = (stft_loss)
            
            loss.backward()
            
            
            # torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter > 1000:
                print(f"Early stopping at iteration {n}")
                break

            pbar.set_postfix({
                "loss": loss.item(), 
                "lr": self.optimizer.param_groups[0]['lr']
            })
        
        # print the trained parameters
        print(f"Trained Frequencies: {self.eq_parameters_freqs.data}")
        print(f"Trained Gains: {self.eq_parameters_gains.data}")
        print(f"Trained Q: {self.eq_parameters_q.data}")
        print(f"Trained Delays: {self.delays.data}")

    
    def forward(self, input_signal: torch.Tensor):
        # Génération de la réponse en fréquence
        self.eq_parameters_freqs = self.parameter_to_frequency()
        self.eq_parameters_gains = self.parameter_to_gains()
        self.eq_parameters_q     = self.parameter_to_q()

        # print the trained parameters
        print(f"Trained Frequencies: {self.eq_parameters_freqs.data}")
        print(f"Trained Gains: {self.eq_parameters_gains.data}")
        print(f"Trained Q: {self.eq_parameters_q.data}")
        print(f"Trained Delays: {self.delays.data}")
        
        # Correction de la taille de fft_freqs
        # La taille de la FFT pour le filtrage doit être suffisamment grande
        n_fft_filter = len(input_signal.squeeze())
        fft_freqs = torch.fft.fftfreq(n_fft_filter)[:n_fft_filter // 2 + 1] * self.sample_rate

        # Assurez-vous que evaluate_mag_response est compatible
        eq_mag_response_lin = evaluate_mag_response(fft_freqs, self.eq_parameters_freqs, self.eq_parameters_gains, self.eq_parameters_q)
        
        # Création de l'IR
        # La taille de l'IR doit être 2*(longueur de la réponse en fréquence)-2
        ir = torch.fft.fftshift(torch.fft.irfft(eq_mag_response_lin.T), dim = -1)
        window = torch.hann_window(ir.size(-1), periodic=False, device=self.device, dtype=torch.float32).expand_as(ir)
        ir = ir * window

        # Le code de filtrage est correct
        n_fft = len(input_signal.squeeze())
        ir_fft_padded = torch.fft.fft(ir.squeeze(), n=n_fft)
        prediction_freq_domain = torch.fft.fft(input_signal.squeeze(), n=n_fft)
        prediction_freq_domain = prediction_freq_domain * ir_fft_padded

        # Retour au domaine temporel
        prediction = torch.real(torch.fft.ifft(prediction_freq_domain))
    
        return prediction

    def parameter_to_frequency(self):
        freq_params = torch.sigmoid(self.parameters[0].unsqueeze(1).repeat(1, self.num_of_delays))
        return frequency_denormalize(freq_params)

    def parameter_to_gains(self):
        # Use tanh activation for bounded output and better gradients
        gain_params = gain_denormalize(torch.sigmoid(self.parameters[1]))
        return convert_proto_gain_to_delay(gain_params.unsqueeze(1).repeat(1, self.num_of_delays), self.delays, self.sample_rate)
    
    def parameter_to_q(self):
        q_params = torch.sigmoid(self.parameters[2].unsqueeze(1).repeat(1, self.num_of_delays))
        return q_denormalize(q_params)
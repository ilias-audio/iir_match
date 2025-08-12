import torch
from Dataset import Dataloader
from utilities import *
from Filters import evaluate_mag_response, evaluate_sos_response, RBJ_LowShelf, RBJ_Bell, RBJ_HighShelf
from tqdm import tqdm
from scipy import signal
import auraloss
import os

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
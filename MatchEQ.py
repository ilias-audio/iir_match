import torch
from Dataloader import Dataloader
from utilities import frequency_normalize, frequency_denormalize 
from utilities import gain_denormalize, gain_normalize, q_denormalize, q_normalize
from utilities import convert_proto_gain_to_delay
from utilities import indivual_mag_response, evaluate_mag_response
import tqdm


class MatchEQ:
    def __init__(self, RT_Dataset: Dataloader , num_of_iter: int, num_of_bands: int, num_of_delays: int, sample_rate: int, device: str):
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.num_of_bands = num_of_bands
        self.num_of_delays = num_of_delays
        self.min_delay_in_seconds = 0.003
        self.max_delay_in_seconds = 0.3
        self.min_delay_in_samples = int(self.min_delay_in_seconds * self.sample_rate)
        self.max_delay_in_samples = int(self.max_delay_in_seconds * self.sample_rate)
        self.num_of_iter = num_of_iter
        self.delays = torch.sort(torch.randint(self.min_delay_in_samples, self.max_delay_in_samples, (self.num_of_delays, 1), device=self.device), dim=0)
        assert self.delays.shape == (self.num_of_delays, 1), "Delays should be a column vector"
        self.loss_function = torch.nn.functional.mse_loss()
        self.dataset_freqs = RT_Dataset.freqs

        self.init_training_parameters()
        self.convert_dataset_rt_to_responses(RT_Dataset)
        self.setup_optimizer()

    def convert_dataset_rt_to_responses(self, RT_Dataset: Dataloader):
        self.responses_dataset = torch.zeros((RT_Dataset.num_of_rt, len(self.dataset_freqs), self.num_of_delays), device=self.device)
        for i in range(RT_Dataset.num_of_rt):
            self.responses_dataset[i,:,:] = (-60 * self.delays) / (self.sample_rate * RT_Dataset.dataset[:, i])

    def init_training_parameters(self):
        self.parameters = torch.nn.ParameterList()

        freq_values = torch.ones(self.num_of_bands, requires_grad=True, device=self.device, dtype=torch.float32)
        freq_values[0:-2] = torch.logspace(torch.log10(20), torch.log10(16000), self.num_of_bands-2)
        # Make the shelfs centered at 1000 Hz
        freq_values[0] = 1000 
        freq_values[-1] = 1000

        self.parameters.append(frequency_normalize((freq_values)))  # Common Freqs
        self.parameters.append(torch.zeros(self.num_of_bands, requires_grad=True, device=self.device, dtype=torch.float32))  # Common gains
        self.parameters.append(torch.zeros(self.num_of_bands, requires_grad=True, device=self.device, dtype=torch.float32))  # Common Q
        
        assert self.parameters[0].shape == (self.num_of_bands, 1), "Frequency values should be a column vector"
        assert self.parameters[1].shape == (self.num_of_bands, 1), "Gain values should be a column vector"
        assert self.parameters[2].shape == (self.num_of_bands, 1), "Q values should be a column vector"
    
    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters, lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.num_of_iter)

    def train(self, dataset_index: int):
        pbar = tqdm(range(self.num_of_iter), desc=f"RT Index {rt_index}")

        for n in pbar:
            self.optimizer.zero_grad()

            pred_response = self.calculate_predicted_response()

            loss = self.loss_function(self.responses_dataset[dataset_index, :,:], pred_response)

            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            pbar.set_postfix({"loss": loss.item()})


    def calculate_predicted_response(self):
        self.eq_parameters_freqs = self.parameter_to_frequency()
        self.eq_parameters_gains = self.parameter_to_gains()
        self.eq_parameters_q     = self.parameter_to_q()

        assert self.eq_parameters_freqs.shape == (self.num_of_bands, self.num_of_delays), "Frequency values should Bands X Delays Matrix"
        assert self.eq_parameters_gains.shape == (self.num_of_bands, self.num_of_delays), "Gain values should Bands X Delays Matrix"
        assert self.eq_parameters_q.shape == (self.num_of_bands, self.num_of_delays), "Q values should Bands X Delays Matrix"

        pred_responses = evaluate_mag_response(self.dataset_freqs, self.eq_parameters_freqs, self.eq_parameters_gains, self.eq_parameters_q)
        
        assert pred_responses.shape == (self.dataset_freqs, self.num_of_delays), "Predicted responses should Freqs X Delays Matrix"
        
        return pred_responses

    def parameter_to_frequency(self):
        freq_params = torch.sigmoid(self.parameters[0].unsqueeze(0).repeat(self.num_of_delays, 1))
        return frequency_denormalize(freq_params)

    def parameter_to_gains(self):
        gain_params = torch.sigmoid(self.parameters[1])
        convert_proto_gain_to_delay(gain_denormalize((gain_params)), self.delays, self.sample_rate)
        return gain_params
    
    def parameter_to_q(self):
        q_params = torch.sigmoid(self.parameters[2].unsqueeze(0).repeat(self.num_of_delays, 1))
        return q_denormalize(q_params)
    
    
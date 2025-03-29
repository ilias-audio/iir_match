import torch
from Dataloader import Dataloader
from utilities import frequency_normalize, frequency_denormalize 
from utilities import gain_denormalize, gain_normalize, q_denormalize, q_normalize
from utilities import convert_proto_gain_to_delay
from Filters import evaluate_mag_response
from tqdm import tqdm


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
        self.delays, _ = torch.sort(torch.randint(self.min_delay_in_samples, self.max_delay_in_samples, (self.num_of_delays, 1), device=self.device, dtype=torch.float32), dim=0)
        assert self.delays.shape == (self.num_of_delays, 1), "Delays should be a column vector"
        self.loss_function = torch.nn.MSELoss()
        self.dataset_freqs = torch.tensor(RT_Dataset.freqs, device=self.device, dtype=torch.float32)

        self.init_training_parameters()
        self.convert_dataset_rt_to_responses(RT_Dataset)
        self.setup_optimizer()

    def convert_dataset_rt_to_responses(self, RT_Dataset: Dataloader):
        self.responses_dataset = torch.zeros((RT_Dataset.num_of_rt, len(self.dataset_freqs), self.num_of_delays), device=self.device)
        for i in range(RT_Dataset.num_of_rt):
            self.responses_dataset[i,:,:] = ((-60 * self.delays.cpu()) / (self.sample_rate * RT_Dataset.dataset[:, i])).T

    def init_training_parameters(self):
        self.parameters = torch.nn.ParameterList()
        init_min_freq = torch.log(torch.tensor(20.))
        init_max_freq = torch.log(torch.tensor(16000.))

        freq_values = torch.ones(self.num_of_bands, requires_grad=False, device=self.device, dtype=torch.float32)
        freq_values[0:-2] = torch.logspace(init_min_freq, init_max_freq, self.num_of_bands-2)
        # Make the shelfs centered at 1000 Hz
        freq_values[0] = 1000 
        freq_values[-1] = 1000

        self.parameters.append(frequency_normalize(freq_values))  # Common Freqs
        self.parameters.append(torch.zeros(self.num_of_bands, requires_grad=True, device=self.device, dtype=torch.float32))  # Common gains
        self.parameters.append(torch.zeros(self.num_of_bands, requires_grad=True, device=self.device, dtype=torch.float32))  # Common Q
        
        assert self.parameters[0].shape == torch.Size([self.num_of_bands]), f"Frequency values should be a column vector, but got {self.parameters[0].shape}"
        assert self.parameters[1].shape == torch.Size([self.num_of_bands]), f"Gain values should be a column vector, but got {self.parameters[1].shape}"
        assert self.parameters[2].shape == torch.Size([self.num_of_bands]), f"Q values should be a column vector, but got {self.parameters[2].shape}"
    
    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters, lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.num_of_iter)

    def train(self, dataset_index: int):
        pbar = tqdm(range(self.num_of_iter), desc=f"RT Index {dataset_index}")

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
        
        assert pred_responses.shape == (len(self.dataset_freqs), self.num_of_delays), "Predicted responses should Freqs X Delays Matrix"
        
        return pred_responses

    def parameter_to_frequency(self):
        freq_params = torch.sigmoid(self.parameters[0].unsqueeze(1).repeat(1, self.num_of_delays))
        return frequency_denormalize(freq_params)

    def parameter_to_gains(self):
        gain_params = torch.sigmoid(self.parameters[1])
        return convert_proto_gain_to_delay(gain_denormalize((gain_params)), self.delays, self.sample_rate)
    
    def parameter_to_q(self):
        q_params = torch.sigmoid(self.parameters[2].unsqueeze(1).repeat(1, self.num_of_delays))
        return q_denormalize(q_params)
    
    
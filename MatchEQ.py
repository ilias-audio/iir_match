import torch
from utilities import *
from tqdm import tqdm
import Filters
import Dataset
import os


class MatchEQ():
    def __init__(self, dataset: Dataset.Dataset,  parameters: dict):
        self.set_sample_rate(parameters["SAMPLE_RATE"])
        self.set_device(parameters["DEVICE"])
        self.set_fft_size(parameters["FFT_SIZE"])
        self.set_number_of_delays(parameters["NUM_OF_DELAYS"])
        self.set_number_of_iterations(parameters["NUM_OF_ITER"])
        self.num_of_bands = parameters["NUM_OF_BANDS"]
        self.set_batch_size(parameters["BATCH_SIZE"])

        self.default_parameters()
        self.set_training_parameters()
        self.setup_optimizer()
        self.loss_function = torch.nn.MSELoss()
        
        self.dataset_freqs = torch.tensor(dataset.freqs, device=self.device, dtype=torch.float32)
        self.dataset = torch.tensor(dataset.dataset, dtype=torch.float32)
        self.convert_dataset_rt_to_responses(dataset)
        # Create the DataLoader with a batch size of your choice
        responses_tensor_dataset = torch.utils.data.TensorDataset(self.responses_dataset.unsqueeze(-1))
        self.dataloader = torch.utils.data.DataLoader(responses_tensor_dataset, batch_size=self.batch_size,shuffle=True)
   
    def default_parameters(self):
        self.set_min_delay_in_seconds(0.003)
        self.set_max_delay_in_seconds(0.1)
        # self.set_random_delays()
        self.set_fixed_delays([4800])
        self.set_min_frequency(1.)
        self.set_max_frequency(20000.)

    def convert_dataset_rt_to_responses(self, dataset: Dataset.Dataset):
        self.median_response = ((-60 * self.delays.cpu()) / (self.sample_rate * dataset.median_rt))
        
        self.responses_dataset = torch.zeros((dataset.num_of_rt, len(self.dataset_freqs), self.num_of_delays), device=self.device)
        for i in range(dataset.num_of_rt):
            self.responses_dataset[i,:,:] = ((-60 * self.delays.cpu()) / (self.sample_rate * dataset.dataset[:, i])).T

    def set_training_parameters(self):
        self.parameters = torch.nn.ParameterList()

        self.parameters.append(self.initial_training_frequencies())
        self.parameters.append(self.initial_training_gains())
        self.parameters.append(self.initial_training_q())

        assert self.parameters[0].shape == torch.Size([self.num_of_bands]), f"Frequency values should be a column vector, but got {self.parameters[0].shape}"
        assert self.parameters[1].shape == torch.Size([self.num_of_bands]), f"Gain values should be a column vector, but got {self.parameters[1].shape}"
        assert self.parameters[2].shape == torch.Size([self.num_of_bands]), f"Q values should be a column vector, but got {self.parameters[2].shape}"
     
    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters, lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.num_of_iter)

    def parameter_to_frequency(self):
        freq_params = torch.sigmoid(self.parameters[0].unsqueeze(1).repeat(1, self.num_of_delays))
        return frequency_denormalize(freq_params)

    def parameter_to_gains(self):
        gain_params = torch.sigmoid(self.parameters[1])
        return convert_proto_gain_to_delay(gain_denormalize((gain_params)), self.delays, self.sample_rate)
    
    def parameter_to_q(self):
        q_params = torch.sigmoid(self.parameters[2].unsqueeze(1).repeat(1, self.num_of_delays))
        return q_denormalize(q_params)




    def initial_training_frequencies(self):
        freq_values = torch.ones(self.num_of_bands, requires_grad=False, device=self.device, dtype=torch.float32)
        freq_values[1:-1] = torch.logspace(self.min_freq, self.max_freq, self.num_of_bands-2)
        # Make the shelfs centered at 1000 Hz
        freq_values[0] = 1000 
        freq_values[-1] = 1000
        # freq_values.requires_grad(True)
        freq_values.requires_grad_(True)
        return torch.special.logit(frequency_normalize(freq_values),eps=1e-6)

    def initial_training_gains(self):
        gain_values = 0.1 * torch.ones(self.num_of_bands, requires_grad=True, device=self.device, dtype=torch.float32)
        return torch.special.logit(gain_normalize(gain_values),eps=1e-6)

    def initial_training_q(self):
        q_values =  0.7 * torch.ones(self.num_of_bands, requires_grad=True, device=self.device, dtype=torch.float32)
        return torch.special.logit(q_normalize(q_values),eps=1e-6)
    


    def calculate_predicted_response(self):
        self.eq_parameters_freqs = self.parameter_to_frequency()
        self.eq_parameters_gains = self.parameter_to_gains().T
        self.eq_parameters_q     = self.parameter_to_q()

        assert self.eq_parameters_freqs.shape == (self.num_of_bands, self.num_of_delays), "Frequency values should Bands X Delays Matrix"
        assert self.eq_parameters_gains.shape == (self.num_of_bands, self.num_of_delays), "Gain values should Bands X Delays Matrix"
        assert self.eq_parameters_q.shape == (self.num_of_bands, self.num_of_delays), "Q values should Bands X Delays Matrix"

        pred_responses = Filters.evaluate_mag_response(self.dataset_freqs, self.eq_parameters_freqs, self.eq_parameters_gains, self.eq_parameters_q)
        
        assert pred_responses.shape == (len(self.dataset_freqs), self.num_of_delays), "Predicted responses should Freqs X Delays Matrix"
        
        return pred_responses

    
    def curve_train(self, dataset_index: int):
        pbar = tqdm(range(self.num_of_iter), desc=f"RT Index {dataset_index}")

        for iter in pbar:
            self.optimizer.zero_grad()

            pred_response = self.calculate_predicted_response()
            pred_response_dB = 20. * torch.log10(pred_response)
  
            loss = self.loss_function(self.responses_dataset[dataset_index, :], pred_response_dB)

            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            pbar.set_postfix({"loss": loss.item()})
        
        # print the trained parameters
        print(f"Trained Frequencies: {self.eq_parameters_freqs.data}")
        print(f"Trained Gains: {self.eq_parameters_gains.data}")
        print(f"Trained Q: {self.eq_parameters_q.data}")
        print(f"Trained Delays: {self.delays.data}")

        self.save_trained_parameters(dataset_index)


    def audio_train(self, target_x: torch.Tensor, input_signal: torch.Tensor):
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

        

    def audio_forward(self, input_signal: torch.Tensor):
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


    def save_trained_parameters(self, dataset_index: int):
        # Create a folder for the dataset_index if it doesn't exist
        folder_name = os.path.join("results", f"dataset_index_{dataset_index}_{self.num_of_bands}_bands")
        os.makedirs(folder_name, exist_ok=True)

        # Save the trained parameters in the folder
        torch.save(self.eq_parameters_freqs, os.path.join(folder_name, f"trained_frequencies_{dataset_index}.pt"))
        torch.save(self.eq_parameters_gains, os.path.join(folder_name, f"trained_gains_{dataset_index}.pt"))
        torch.save(self.eq_parameters_q, os.path.join(folder_name, f"trained_q_{dataset_index}.pt"))
        torch.save(self.delays, os.path.join(folder_name, f"trained_delays_{dataset_index}.pt"))

        pred_response_dB = 20. * torch.log10(self.calculate_predicted_response().detach().cpu())

        pred_rt = convert_response_to_rt(pred_response_dB, self.delays, self.sample_rate)

        torch.save(self.dataset_freqs, os.path.join(folder_name, f"pred_freqs_{dataset_index}.pt"))
        torch.save(pred_response_dB, os.path.join(folder_name, f"pred_response_{dataset_index}.pt"))
        torch.save(pred_rt, os.path.join(folder_name, f"pred_rt_{dataset_index}.pt"))



    # Methods for FFT
    def set_fft_size(self, fft_size):
        self._fft_size = fft_size

    # Methods for delays 
    def set_number_of_delays(self, num_of_delays):
        self._num_of_delays = num_of_delays

    def set_fixed_delays(self, delays):
        self.set_number_of_delays = len(delays)
        self.delays = torch.tensor(delays, device=self.device, dtype=torch.float32).repeat(self.num_of_delays, 1)
        assert self.delays.shape == (self.num_of_delays, 1), "Delays should be a column vector"

    def set_random_delays(self):
        self.delays, _ = torch.sort(torch.randint(self.min_delay_in_samples, self.max_delay_in_samples, (self.num_of_delays, 1), device=self.device, dtype=torch.float32), dim=0)

    # Max
    def set_max_delay_in_seconds(self, delay : float):
        self._max_delay_in_seconds = delay

    # Min
    def set_min_delay_in_seconds(self, delay : float):
        self._min_delay_in_seconds = delay

    # Methods for sampling rate
    def set_sample_rate(self, sample_rate :int):
        self._sample_rate = sample_rate

    # Methods for pytorch device
    def set_device(self, device : str):
        self._device = torch.device(device)

    def set_batch_size(self, bs):
        self._batch_size = bs

    def set_number_of_iterations(self, num_of_iter):
        self._num_of_iter = num_of_iter

        # Methods for frequencies
    def set_max_frequency(self, freq : float):
        self._max_frequency = freq

    def set_min_frequency(self, freq : float):
        self._min_frequency = freq
    
    @property
    def min_freq(self):
        return self._min_frequency

    @property
    def max_freq(self):
        return self._max_frequency

    @property
    def num_of_delays(self):
        return self._num_of_delays

    @property
    def num_of_iter(self):
        return self._num_of_iter
    
    @property 
    def batch_size(self):
        return self._batch_size
    
    @property
    def fft_size(self):
        return self._fft_size
    
    @property
    def device(self):
        return self._device
    
    @property
    def sample_rate(self):
        return self._sample_rate
    
    @property
    def min_delay_in_seconds(self):
        return self._min_delay_in_seconds

    @property
    def min_delay_in_samples(self):
        return self._min_delay_in_seconds * self.sample_rate
    
    @property
    def max_delay_in_seconds(self):
        return self._max_delay_in_seconds

    @property
    def max_delay_in_samples(self):
        return self._max_delay_in_seconds * self.sample_rate
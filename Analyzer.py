import os
import torch

class Analyzer:
    def __init__(self, folder_pattern: str):
        self.folder_pattern = folder_pattern
        self.number_of_responses = 0
        self.number_of_delays = 0
        self.trained_responses = None
        self.trained_frequencies = None
    
    def load_responses(self, number_of_responses: int, number_of_delays: int):
        self.number_of_responses = number_of_responses
        self.number_of_delays = number_of_delays

        freq_filename = os.path.join(self.folder_pattern + str(0), "pred_freqs_" + str(0) + ".pt")
        assert os.path.exists(freq_filename), f"File {freq_filename} does not exist."
        
        # Load the frequency response
        trained_frequencies = torch.load(freq_filename)
        assert trained_frequencies.ndim == 1, "Frequency response should be one-dimensional"
        self.trained_frequencies = trained_frequencies

        self.trained_responses = torch.zeros((number_of_responses, self.trained_frequencies.shape[0], self.number_of_delays), device=self.trained_frequencies.device)
        
        for i in range(number_of_responses):
            resp_filename = os.path.join(self.folder_pattern + str(i), "pred_response_" + str(i) + ".pt")
            assert os.path.exists(resp_filename), f"File {resp_filename} does not exist."
            
            self.trained_responses[i,:,:] = torch.load(resp_filename)

    
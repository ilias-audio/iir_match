import os
import torch
import Dataloader

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
        self.trained_rt = torch.zeros((number_of_responses, self.trained_frequencies.shape[0], self.number_of_delays), device=self.trained_frequencies.device)
        
        for i in range(number_of_responses):
            resp_filename = os.path.join(self.folder_pattern + str(i), "pred_response_" + str(i) + ".pt")
            assert os.path.exists(resp_filename), f"File {resp_filename} does not exist."
            rt_filename = os.path.join(self.folder_pattern + str(i), "pred_rt_" + str(i) + ".pt")
            assert os.path.exists(resp_filename), f"File {rt_filename} does not exist."
            
            self.trained_responses[i,:,:] = torch.load(resp_filename)
            self.trained_rt[i,:,:] = torch.load(rt_filename)
    
    def compute_relative_error(self, RT_Dataset: Dataloader):
        self.relative_error = torch.zeros((self.trained_rt.shape[0:2]), device=self.trained_responses.device)
        for i in range(self.trained_rt.shape[0]):
            self.relative_error[i,:] = ((torch.tensor(RT_Dataset.dataset[:,i]) - self.trained_rt[i,:,0]) / RT_Dataset.dataset[:,i]) * 100
        import matplotlib.pyplot as plt
        plt.clf()
        plt.hist(self.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=RT_Dataset.dataset.shape[1] // 100)
        plt.xlabel("Percentage Error")
        plt.ylabel("Probability")
        plt.title("T_{60} Error Distribution")
        plt.savefig(os.path.join("figures", "figure_4.png"))
    

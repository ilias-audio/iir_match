import os
import torch
import Dataset
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D

# font = {'family' : 'serif', 
#         'size'   : 20}

# plt.rc('font', **font)

class Analyzer:
    def __init__(self):
        os.makedirs("figures", exist_ok=True)
        self._fig_size = (10, 8)
        self.folder_pattern = "results/dataset_index_"

    def fig_dataset_rt_dist(self, dataset):
        dataset_size = len(dataset.dataset.flatten())
        weights = np.ones(dataset_size) / dataset_size * 100

        plt.figure(figsize=self._fig_size)
        plt.hist(dataset.dataset.flatten(), weights=weights)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '%1.0f%%' % x))
        plt.xlabel("Reverberation Time (s)")
        plt.ylabel("Percentage")
        plt.title("Dataset Reverberation Times distribution")
        plt.savefig(os.path.join("figures", "dataset_rt_distribution.png"))
        plt.close()

    def fig_dataset_rt_boxplot(self, dataset):
        '''Box Plot of RT values'''
        plt.figure(figsize=self._fig_size)
        plt.boxplot(dataset.raw_dataset.T)
        
        x_ticks = [f"{int(freq/1000)}k" if freq > 950 else str(int(freq)) for freq in dataset.raw_dataset_freqs]
        x_ticks.insert(0, "")  # Insert an empty string as the first xtick
        plt.xticks(range(len(dataset.raw_dataset_freqs) + 1), x_ticks, fontsize=8)  # Adjust the range accordingly
        plt.title("Reverberation Time Values for Each Frequency")
        plt.savefig(os.path.join("figures", "dataset_rt_boxplot.png"))
        # this should ideally be 1 per octave on the X-axis
        plt.close()

    def fig_check_interp(self, rt_index, dataset):
        plt.figure(figsize=self._fig_size)
        plt.semilogx(dataset.freqs, dataset.dataset[:,rt_index])
        plt.semilogx(dataset.raw_dataset_freqs, dataset.raw_dataset[:,rt_index], 'o')
        plt.savefig(os.path.join("figures", f"interpolation_check_rt_{rt_index}.png"))
        plt.close()
    
    def fig_check_median_rt(self, dataset):
        plt.figure(figsize=self._fig_size)
        raw_median_rt = np.median(dataset.raw_dataset, axis=1)
        plt.semilogx(dataset.raw_dataset_freqs, raw_median_rt, 'o', label="Raw Median")
        plt.semilogx(dataset.freqs, dataset.median_rt, label="Interpolate Median")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Median Dataset T$_{60}$ (s)")
        plt.title("Median RT values for each frequency")
        plt.savefig(os.path.join("figures", "median_rt.png"))
        plt.legend()
        plt.close()


    def load_responses(self, number_of_eq_bands: int, parameters: dict):
        number_of_responses = parameters["NUM_OF_RT"]
        number_of_delays = parameters["NUM_OF_DELAYS"]
        device = parameters["DEVICE"]


        freq_filename = os.path.join(self.folder_pattern + str(0) + "_" + str(number_of_eq_bands) + "_bands", "pred_freqs_" + str(0) + ".pt")
        assert os.path.exists(freq_filename), f"File {freq_filename} does not exist."
        
        # Load the frequency response
        trained_frequencies = torch.load(freq_filename)
        assert trained_frequencies.ndim == 1, "Frequency response should be one-dimensional"
        self.trained_frequencies = trained_frequencies

        self.trained_responses = torch.zeros((number_of_responses, self.trained_frequencies.shape[0], number_of_delays), device=device)
        self.trained_rt = torch.zeros((number_of_responses, self.trained_frequencies.shape[0], number_of_delays), device=device)
        
        for i in range(number_of_responses):
            folder_name = self.folder_pattern + str(i) + "_" + str(number_of_eq_bands) + "_bands"
            resp_filename = os.path.join(folder_name, "pred_response_" + str(i) + ".pt")
            assert os.path.exists(resp_filename), f"File {resp_filename} does not exist."
            rt_filename = os.path.join(folder_name, "pred_rt_" + str(i) + ".pt")
            assert os.path.exists(resp_filename), f"File {rt_filename} does not exist."
            
            self.trained_responses[i,:,:] = torch.load(resp_filename)
            self.trained_rt[i,:,:] = torch.load(rt_filename)
    
    def compute_relative_error(self, num_of_eq_bands: int, RT_Dataset: Dataset, parameters: dict):
        self.load_responses(num_of_eq_bands, parameters)
        number_of_RT = parameters["NUM_OF_RT"]
        # import matplotlib as mpl
        # mpl.rcParams.update({
        #     "text.usetex": True,
        #     "font.family": "serif",
        #     "font.serif": ["Times"],
        #     "axes.labelsize": 10,
        #     "xtick.labelsize": 10,
        #     "ytick.labelsize": 10,
        # })
        # # plt.tight_layout(rect=[0, 0, 1, 0.8])
        # mpl.rcParams.update({
        #     "ytick.minor.visible": False,
        #     "ytick.major.size": 5,
        #     "ytick.labelsize": 10,
        #     "ytick.major.left": True,
        #     "ytick.labelleft": True,
        # })
        self.relative_error = torch.zeros((self.trained_rt.shape[0:2]), device=self.trained_responses.device)
        start = 0
        for i in range(self.trained_rt.shape[0]):
            self.relative_error[i,start:] = ((torch.tensor(RT_Dataset.dataset[start:,i]) - self.trained_rt[i,start:,0]) / RT_Dataset.dataset[start:,i]) * 100
        

        torch.save(self.relative_error, os.path.join("results", f"dataset_relative_error_{num_of_eq_bands}_bands.pt"))
        # plt.figure(figsize=(4, 2.5))
        
        # plt.clf()
        # # plt.subplot(1, 1, 1)
        # plt.hist(self.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=RT_Dataset.num_of_rt // 10)
        # plt.xlabel(r"T$_{60}$ Error (\%)")
        # plt.ylabel("Probability")
        # plt.xlim((-110,110))
        # plt.ylim((1e-6, 1e0))
        # plt.xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100], labels=["$-100$", "$-75$", "$-50$", "$-25$", "$0$", "$25$", "$50$", "$75$", "$100$"])
        # plt.yticks([ 1e-4, 1e-2, 1e0], labels=["$10^{-4}$", "$10^{-2}$", "$1$"])
        # # plt.title("T60 Error Distribution")
        # plt.tight_layout()
        # plt.savefig(os.path.join("figures", "figure_6.png"), dpi=300, bbox_inches='tight')
        # plt.close()
    
    def fig_relative_error(self):
        # find_number_of_relative_error_pt_files
        filename_header = "dataset_relative_error_"
        relative_error_files = [f for f in os.listdir("results") if f.startswith(filename_header) and f.endswith("_bands.pt")]
        print(f"Found {len(relative_error_files)} relative error files.")
        print(relative_error_files)
        relative_error_files.sort(key=lambda x: int(x.split('_')[3]))  # Sort by number of bands
        print(f"Sorted relative error files: {relative_error_files}")
        


        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })
        # plt.tight_layout(rect=[0, 0, 1, 0.8])
        mpl.rcParams.update({
            "ytick.minor.visible": False,
            "ytick.major.size": 5,
            "ytick.labelsize": 10,
            "ytick.major.left": True,
            "ytick.labelleft": True,
        })
        legend_elements = []
        plt.figure()
        ax = plt.gca()
        for i, file in enumerate(relative_error_files):
            error_data = torch.load(os.path.join("results", file))
            is_last = i == len(relative_error_files) - 1
            linestyle = 'solid' if is_last else 'dotted'
            color = 'red' if is_last else None
            linewidth = 2.5 if is_last else 1.0

            # Plot and get the color used by matplotlib
            n, bins, patches = ax.hist(
            error_data.cpu().numpy().flatten(),
            density=True,
            histtype='step',
            log=True,
            bins=error_data.shape[1] // 10,
            label=f"{file.split('_')[3]} bands",
            linestyle=linestyle,
            color=color,
            linewidth=linewidth
            )
            # Get the color from the first patch (Line2D object)
            legend_color = color if color else (patches[0].get_edgecolor() if patches else 'black')
            legend_elements.append(Line2D([0], [0], color=legend_color, linestyle=linestyle, linewidth=linewidth, label=f"{file.split('_')[3]} bands"))
        # error_data = torch.load(os.path.join("results", relative_error_files[0]))
        
        # plt.hist(Results_Analyzer_4_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT // 10, label=f"{4} bands", linestyle='dotted')
        # # plt.hist(Results_Analyzer_6_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT, label=f"{4} bands")
        # plt.hist(Results_Analyzer_8_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT // 10, label=f"{8} bands", linestyle='dotted')
        # # plt.hist(Results_Analyzer_10_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT, label=f"{4} bands")
        # plt.hist(Results_Analyzer_12_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT // 10, label=f"{12} bands", color='red')
        plt.xlabel(r"T$_{60}$ Error (\%)")
        plt.ylabel("Probability")
        plt.xlim((-110,110))
        plt.ylim((1e-6, 1e0))
        plt.legend()
        plt.xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100], labels=["$-100$", "$-75$", "$-50$", "$-25$", "$0$", "$25$", "$50$", "$75$", "$100$"])
        plt.yticks([ 1e-4, 1e-2, 1e0], labels=["$10^{-4}$", "$10^{-2}$", "$1$"])
        

        # legend_elements = [
        #     Line2D([0], [0], color='blue', linestyle='dotted', label='4 bands'),
        #     Line2D([0], [0], color='orange', linestyle='dotted', label='8 bands'),
        #     Line2D([0], [0], color='red', linestyle='solid', label='12 bands'),
        # ]

        # plt.legend(handles=legend_elements)
        plt.legend(handles=legend_elements)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "figure_4.png"),dpi=300, bbox_inches='tight')
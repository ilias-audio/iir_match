import os
import Dataloader
import MatchEQ
import Analyzer
import multiprocessing
import concurrent.futures
import torch
import matplotlib.pyplot as plt

import numpy as np


if __name__ == "__main__":

    ######### TOP LEVEL PARAMETERS ###########
    SAMPLE_RATE = 48000
    NUM_OF_DELAYS = 1
    NUM_OF_BANDS = 31
    NUM_OF_ITER = 10000
    INTERPOLATION_SIZE = 512
    NUM_OF_RT = 1000
    ##########################################

    # Load dataset and interpolate data to a bigger size
    raw_dataset_path = os.path.join("data", "two-stage-RT-values.mat")
    RT_dataset = Dataloader.Dataloader(raw_dataset_path, INTERPOLATION_SIZE)

    # print(RT_dataset.median_rt.shape)
    # print(RT_dataset.raw_median_rt)

    def train_median(number_of_bands :int):
        EQ_Optimizer = MatchEQ.MatchEQ(RT_dataset, NUM_OF_ITER, number_of_bands, NUM_OF_DELAYS, SAMPLE_RATE, "cpu")
        EQ_Optimizer.train_median()
        EQ_Optimizer.plot_median_results()
        EQ_Optimizer.save_median_parameters()

    max_workers = max(1, multiprocessing.cpu_count()-2)
    print(f"Using {max_workers} CPU workers for training.")




    plt.clf()
    plt.figure(figsize=(4, 2.5))
    import matplotlib as mpl
    import matplotlib.pyplot as plt
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

    target_response = (-60 * 4800) / (48000 * RT_dataset.median_rt)
    
    for i in [4,8,12]:
        # train_median(i)


        response_4_bands = torch.load(f"results/dataset_median_{i}_bands/pred_response_median.pt")
        
        mse = np.mean((target_response.squeeze() - response_4_bands.detach().cpu().numpy().squeeze()) ** 2)
        mae = np.max(abs(target_response.squeeze() - response_4_bands.detach().cpu().numpy().squeeze()))

        plt.semilogx(RT_dataset.freqs, response_4_bands.squeeze(), label=f"PEQ$_{{{i}}}$", linestyle="--")
        print()
        print(f"MSE and MAE for {i} bands:")
        print("Mean Squared Error =     ",f"{mse:.1e}")
        print("Maximum Absolute Error = ", f"{mae:.1e}")
        print()
        #

    plt.semilogx(RT_dataset.freqs, target_response.squeeze(), label="Target", color="black", linestyle="--")
    
    plt.legend()
    plt.xticks([1, 10, 30, 100, 300, 1e3,3e3, 1e4, 2e4], ['1', '10', '30','100', '300', '1k', '3k','10k','20k'])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude Response(dB)")
    plt.tight_layout()
    plt.savefig(f"results/target_response_median.png", dpi=300, bbox_inches='tight')

    



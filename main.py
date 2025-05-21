import os
import Dataloader
import MatchEQ
import Analyzer
import multiprocessing
import concurrent.futures




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
    raw_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "two-stage-RT-values.mat")
    RT_dataset = Dataloader.Dataloader(raw_dataset_path, INTERPOLATION_SIZE)

    # print(RT_dataset.median_rt.shape)
    print(RT_dataset.raw_median_rt)

    # Run the process in parallel so it doesn't take forever to train on 1000 RTs
    def train_instance(i):
        EQ_Optimizer = MatchEQ.MatchEQ(RT_dataset, NUM_OF_ITER, NUM_OF_BANDS, NUM_OF_DELAYS, SAMPLE_RATE, "cpu")
        EQ_Optimizer.train(i)
        EQ_Optimizer.plot_training_results(i)
        EQ_Optimizer.save_trained_parameters(i)

    def train_median(number_of_bands :int):
        EQ_Optimizer = MatchEQ.MatchEQ(RT_dataset, NUM_OF_ITER, number_of_bands, NUM_OF_DELAYS, SAMPLE_RATE, "cpu")
        EQ_Optimizer.train_median()
        EQ_Optimizer.plot_median_results()
        EQ_Optimizer.save_median_parameters()

    max_workers = max(1, multiprocessing.cpu_count()-2)
    print(f"Using {max_workers} CPU workers for training.")

    for i in [4,8,12,24,31]:
        train_median(i)

    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     try:
    #         results = executor.map(train_instance, range(NUM_OF_RT))
    #         for result in results:
    #             print(result)
    #     except Exception as e:
    #         print("Caught an exception during parallel training:", e)
    
    # train_instance(0)
    # train_instance(1)

    def results():
    # load saved responses with freqs on all responses
        Results_Analyzer_4_bands = Analyzer.Analyzer(os.path.join("results", "dataset_"), 4)
        Results_Analyzer_4_bands.load_responses(NUM_OF_RT, NUM_OF_DELAYS)
        Results_Analyzer_4_bands.compute_relative_error(RT_dataset, NUM_OF_RT)

        Results_Analyzer_6_bands = Analyzer.Analyzer(os.path.join("results", "dataset_"), 6)
        Results_Analyzer_6_bands.load_responses(NUM_OF_RT, NUM_OF_DELAYS)
        Results_Analyzer_6_bands.compute_relative_error(RT_dataset, NUM_OF_RT)

        Results_Analyzer_8_bands = Analyzer.Analyzer(os.path.join("results", "dataset_"), 8)
        Results_Analyzer_8_bands.load_responses(NUM_OF_RT, NUM_OF_DELAYS)
        Results_Analyzer_8_bands.compute_relative_error(RT_dataset, NUM_OF_RT)

        Results_Analyzer_10_bands = Analyzer.Analyzer(os.path.join("results", "dataset_"), 10)
        Results_Analyzer_10_bands.load_responses(NUM_OF_RT, NUM_OF_DELAYS)
        Results_Analyzer_10_bands.compute_relative_error(RT_dataset, NUM_OF_RT)

        Results_Analyzer_12_bands = Analyzer.Analyzer(os.path.join("results", "dataset_"), 12)
        Results_Analyzer_12_bands.load_responses(NUM_OF_RT, NUM_OF_DELAYS)
        Results_Analyzer_12_bands.compute_relative_error(RT_dataset, NUM_OF_RT)

        print(Results_Analyzer_12_bands.relative_error)

        

        # compare the trained responses to the RT_dataset
        #reduce dataset to 31 values 


        import matplotlib.pyplot as plt
        import numpy as np

        # data = np.interp(RT_dataset.dataset_freqs, RT_dataset.freqs, Results_Analyzer.trained_rt[0,:,0].cpu().numpy())
        plt.clf()

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
        plt.hist(Results_Analyzer_4_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT // 10, label=f"{4} bands", linestyle='dotted')
        # plt.hist(Results_Analyzer_6_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT, label=f"{4} bands")
        plt.hist(Results_Analyzer_8_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT // 10, label=f"{8} bands", linestyle='dotted')
        # plt.hist(Results_Analyzer_10_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT, label=f"{4} bands")
        plt.hist(Results_Analyzer_12_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT // 10, label=f"{12} bands", color='red')
        plt.xlabel(r"T$_{60}$ Error (\%)")
        plt.ylabel("Probability")
        plt.xlim((-110,110))
        plt.ylim((1e-6, 1e0))
        plt.xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100], labels=["$-100$", "$-75$", "$-50$", "$-25$", "$0$", "$25$", "$50$", "$75$", "$100$"])
        plt.yticks([ 1e-4, 1e-2, 1e0], labels=["$10^{-4}$", "$10^{-2}$", "$1$"])
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color='blue', linestyle='dotted', label='4 bands'),
            Line2D([0], [0], color='orange', linestyle='dotted', label='8 bands'),
            Line2D([0], [0], color='red', linestyle='solid', label='12 bands'),
        ]

        plt.legend(handles=legend_elements)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "figure_4.png"),dpi=300, bbox_inches='tight')





    
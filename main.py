import os
import Dataset
import MatchEQ
import Analyzer
import multiprocessing




if __name__ == "__main__":

    ######### TOP LEVEL PARAMETERS ###########

    parameters = {
        "SAMPLE_RATE": 48000,
        "NUM_OF_DELAYS": 1,
        "NUM_OF_BANDS": 12,
        "NUM_OF_ITER": 10,
        "INTERPOLATION_SIZE": 512,
        "FFT_SIZE": 256,
        "NUM_OF_RT": 3, 
        "DEVICE": "cpu"
    }
    ##########################################

    # Load dataset and interpolate data to a bigger size
    dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "two-stage-RT-values.mat")
    dataset = Dataset.Dataset(dataset_path, parameters)

    analyzer = Analyzer.Analyzer()
    analyzer.fig_dataset_rt_dist(dataset)
    analyzer.fig_dataset_rt_boxplot(dataset)
    analyzer.fig_check_interp(0, dataset)
    analyzer.fig_check_median_rt(dataset)

    solver = MatchEQ.MatchEQ(dataset, parameters)

    solver.curve_train(0)
    solver.curve_train(1)
    solver.curve_train(2)


    analyzer.compute_relative_error(4, dataset, parameters) # can change the number of eq bands in the check
    analyzer.compute_relative_error(6, dataset, parameters) # can change the number of eq bands in the check
    analyzer.compute_relative_error(8, dataset, parameters) # can change the number of eq bands in the check
    analyzer.compute_relative_error(10, dataset, parameters) # can change the number of eq bands in the check
    analyzer.compute_relative_error(12, dataset, parameters) # can change the number of eq bands in the check
    # but doesn't change the picture name so overwrite figure 4 or figure 6.
    # I could just export the relative error measurments and then load them in a picture to create the final one
    # analyzer.generate_relative_error_figure()
    analyzer.fig_relative_error()


    # Run the process in parallel so it doesn't take forever to train on 1000 RTs
    
    # solver = MatchEQ.MatchEQ(dataset, NUM_OF_ITER, NUM_OF_BANDS, NUM_OF_DELAYS, INTERPOLATION_SIZE, SAMPLE_RATE)
    # solver.curve_train(2)

    


    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     try:
    #         results = executor.map(train_instance, range(NUM_OF_RT))
    #         for result in results:
    #             print(result)
    #     except Exception as e:
    #         print("Caught an exception during parallel training:", e)
    
    # train_instance(0)
    # train_instance(1)

    
        

        

        # compare the trained responses to the RT_dataset
        #reduce dataset to 31 values 


        # data = np.interp(RT_dataset.dataset_freqs, RT_dataset.freqs, Results_Analyzer.trained_rt[0,:,0].cpu().numpy())
        # plt.clf()

        # import matplotlib as mpl
        # import matplotlib.pyplot as plt
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
        # plt.hist(Results_Analyzer_4_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT // 10, label=f"{4} bands", linestyle='dotted')
        # # plt.hist(Results_Analyzer_6_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT, label=f"{4} bands")
        # plt.hist(Results_Analyzer_8_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT // 10, label=f"{8} bands", linestyle='dotted')
        # # plt.hist(Results_Analyzer_10_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT, label=f"{4} bands")
        # plt.hist(Results_Analyzer_12_bands.relative_error.cpu().numpy().flatten(), density=True, histtype='step', log=True, bins=NUM_OF_RT // 10, label=f"{12} bands", color='red')
        # plt.xlabel(r"T$_{60}$ Error (\%)")
        # plt.ylabel("Probability")
        # plt.xlim((-110,110))
        # plt.ylim((1e-6, 1e0))
        # plt.xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100], labels=["$-100$", "$-75$", "$-50$", "$-25$", "$0$", "$25$", "$50$", "$75$", "$100$"])
        # plt.yticks([ 1e-4, 1e-2, 1e0], labels=["$10^{-4}$", "$10^{-2}$", "$1$"])
        # from matplotlib.lines import Line2D

        # legend_elements = [
        #     Line2D([0], [0], color='blue', linestyle='dotted', label='4 bands'),
        #     Line2D([0], [0], color='orange', linestyle='dotted', label='8 bands'),
        #     Line2D([0], [0], color='red', linestyle='solid', label='12 bands'),
        # ]

        # plt.legend(handles=legend_elements)
        # plt.tight_layout()
        # plt.savefig(os.path.join("figures", "figure_4.png"),dpi=300, bbox_inches='tight')





    # results()
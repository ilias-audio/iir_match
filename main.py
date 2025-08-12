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
        "NUM_OF_BANDS": 4,
        "NUM_OF_ITER": 600,
        "INTERPOLATION_SIZE": 512,
        "FFT_SIZE": 256,
        "NUM_OF_RT": 10, 
        "BATCH_SIZE": 4,
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

    bands = [4, 6, 8, 10, 12]
    for num_of_bands in bands:
        parameters["NUM_OF_BANDS"] = num_of_bands
        for i in range(parameters["NUM_OF_RT"]):
            solver = MatchEQ.MatchEQ(dataset, parameters)
            solver.curve_train(i)
        analyzer.compute_relative_error(num_of_bands, dataset, parameters)
    analyzer.fig_relative_error()
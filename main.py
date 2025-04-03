import os
import Dataloader
import MatchEQ
import Analyzer
import multiprocessing
import concurrent.futures

# Run the process in parallel so it doesn't take forever to train on 1000 RTs
def train_instance(i):
    EQ_Optimizer = MatchEQ.MatchEQ(RT_Dataset, NUM_OF_ITER, NUM_OF_BANDS, NUM_OF_DELAYS, SAMPLE_RATE, "cpu")
    EQ_Optimizer.train(i)
    EQ_Optimizer.plot_training_results(i)
    EQ_Optimizer.save_trained_parameters(i)


if __name__ == "__main__":

    ######### TOP LEVEL PARAMETERS ###########
    SAMPLE_RATE = 48000
    NUM_OF_DELAYS = 1
    NUM_OF_BANDS = 12
    NUM_OF_ITER = 10000
    INTERPOLATION_SIZE = 512
    NUM_OF_RT = 1000
    ##########################################

    # Load dataset and interpolate data to a bigger size
    raw_dataset_path = os.path.join("data", "two-stage-RT-values.mat")
    RT_Dataset = Dataloader.Dataloader(raw_dataset_path, INTERPOLATION_SIZE)

    max_workers = max(1, multiprocessing.cpu_count())
    print(f"Using {max_workers} CPU workers for training.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(train_instance, range(NUM_OF_RT))
    
    # train_instance(1)

    
    # load saved responses with freqs on all responses
    Results_Analyzer = Analyzer.Analyzer("dataset_", NUM_OF_BANDS)
    Results_Analyzer.load_responses(NUM_OF_RT, NUM_OF_DELAYS)

    Results_Analyzer.compute_relative_error(RT_Dataset, NUM_OF_RT)

    print(Results_Analyzer.trained_frequencies.shape)
    print(Results_Analyzer.trained_responses.shape)
    print(Results_Analyzer.trained_rt.shape)

    print(Results_Analyzer.trained_rt[0,:,0].shape)
    print(RT_Dataset.dataset[:,0].shape)

    # compare the trained responses to the RT_Dataset
    #reduce dataset to 31 values 


    import matplotlib.pyplot as plt
    import numpy as np

    data = np.interp(RT_Dataset.dataset_freqs, RT_Dataset.freqs, Results_Analyzer.trained_rt[0,:,0].cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.semilogx(RT_Dataset.dataset_freqs, data, label="Trained Response")
    plt.semilogx(RT_Dataset.freqs, RT_Dataset.dataset[:,0], label="RT Dataset")
    plt.legend()
    plt.savefig(f"trained_vs_RT_{0}.png")


    # evaluate each responses at the frequency of the RT_Dataset



    # I need to convert back to RT rather than magnitude and then compare

    # make the relative error of the two 
    # create a historigram of the relative error in the prediction






    
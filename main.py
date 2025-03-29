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
    NUM_OF_ITER = 101
    INTERPOLATION_SIZE = 128
    NUM_OF_RT = 10
    ##########################################

    # Load dataset and interpolate data to a bigger size
    raw_dataset_path = os.path.join("data", "two-stage-RT-values.mat")
    RT_Dataset = Dataloader.Dataloader(raw_dataset_path, INTERPOLATION_SIZE)

    max_workers = max(1, multiprocessing.cpu_count() // 2)
    print(f"Using {max_workers} CPU workers for training.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(train_instance, range(NUM_OF_RT))

    
    # load saved responses with freqs on all responses
    Results_Analyzer = Analyzer.Analyzer("dataset_")
    Results_Analyzer.load_responses(NUM_OF_RT, NUM_OF_DELAYS)

    print(Results_Analyzer.trained_frequencies.shape)
    print(Results_Analyzer.trained_responses.shape)


    # evaluate each responses at the frequency of the RT_Dataset



    # I need to convert back to RT rather than magnitude and then compare

    # make the relative error of the two 
    # create a historigram of the relative error in the prediction






    
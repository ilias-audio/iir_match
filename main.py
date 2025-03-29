import os
import Dataloader
import MatchEQ
import multiprocessing
import concurrent.futures

if __name__ == "__main__":

    ######### TOP LEVEL PARAMETERS ###########
    SAMPLE_RATE = 48000
    NUM_OF_DELAYS = 1
    NUM_OF_BANDS = 4
    NUM_OF_ITER = 1000
    INTERPOLATION_SIZE = 128
    ##########################################

    # Load dataset and interpolate data to a bigger size
    raw_dataset_path = os.path.join("data", "two-stage-RT-values.mat")
    RT_Dataset = Dataloader.Dataloader(raw_dataset_path, INTERPOLATION_SIZE)


    # Run the process in parallel so it doesn't take forever to train on 1000 RTs
    def train_instance(i):
        EQ_Optimizer = MatchEQ.MatchEQ(RT_Dataset, NUM_OF_ITER, NUM_OF_BANDS, NUM_OF_DELAYS, SAMPLE_RATE, "cpu")
        EQ_Optimizer.train(i)
        EQ_Optimizer.plot_training_results(i)

    max_workers = max(1, multiprocessing.cpu_count() // 2)
    print(f"Using {max_workers} CPU workers for training.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(train_instance, range(1000))



    
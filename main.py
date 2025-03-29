import os
import Dataloader
import MatchEQ


if __name__ == "__main__":

    ######### TOP LEVEL PARAMETERS ###########
    SAMPLE_RATE = 48000
    NUM_OF_DELAYS = 100
    NUM_OF_BANDS = 31
    NUM_OF_ITER = 3000
    INTERPOLATION_SIZE = 128
    ##########################################

    # Load dataset and interpolate data to a bigger size
    raw_dataset_path = os.path.join("data", "two-stage-RT-values.mat")
    RT_Dataset = Dataloader.Dataloader(raw_dataset_path, INTERPOLATION_SIZE)

    # Define the optimizer 
    EQ_Optimizer = MatchEQ.MatchEQ(RT_Dataset, NUM_OF_ITER, NUM_OF_BANDS, NUM_OF_DELAYS, SAMPLE_RATE, "cpu")

    EQ_Optimizer.train(0)
    # Save the optimizer state
    EQ_Optimizer.plot_training_results(0)



    
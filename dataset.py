import scipy.io
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'serif', 
        'size'   : 20}

plt.rc('font', **font)


def import_rt_dataset(path: str):
    raw_dataset = scipy.io.loadmat(path)
    return np.array([raw_dataset["rt_"]]).squeeze()
    
def dataset_freq():
    ''' 
    taken directly from Fig_5.m, line 40 on Two_stage_filter repo
    f =  10^3 * (2 .^ ([-17:13]/3))
    '''
    return 10**3 * (2 ** (np.arange(-17, 14) / 3))

def figure_1(rt_dataset, freq_dataset):
    dataset_size = len(rt_dataset.flatten())
    weights = np.ones(dataset_size) / dataset_size * 100

    fig = plt.figure(figsize=(10, 8))  # Adjust the figsize as per your requirement

    plt.hist(rt_dataset.flatten(), weights=weights)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '%1.0f%%' % x))
    plt.xlabel("Reverberation Time (s)")
    plt.ylabel("Percentage")
    plt.title("Histogram of Reverberation Times")
    plt.savefig(os.path.join("figures", "figure_1.png"))

def main():
    print(os.getcwd())
    path = os.path.join("imports", "two-stage-RT-values.mat")
    rt_dataset = import_rt_dataset(path)
    freq_dataset = dataset_freq()
    figure_1(rt_dataset, freq_dataset)
    
    
    
if __name__ == "__main__":
    main()
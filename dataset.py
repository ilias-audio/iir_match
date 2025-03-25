import scipy.io
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate as interp

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

def figure_1(rt_dataset):
    dataset_size = len(rt_dataset.flatten())
    weights = np.ones(dataset_size) / dataset_size * 100

    plt.figure(figsize=(10, 8))

    plt.hist(rt_dataset.flatten(), weights=weights)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '%1.0f%%' % x))
    plt.xlabel("Reverberation Time (s)")
    plt.ylabel("Percentage")
    plt.title("Dataset Reverberation Times distribution")
    plt.savefig(os.path.join("figures", "figure_1.png"))


def interpolate_dataset_response(rt_dataset, freq_dataset, N=24000):
    all_freqs = np.linspace(np.log10(1), np.log10(24000), N)
    print(all_freqs)
    new_dataset = np.zeros((N, rt_dataset.shape[1]))
    for i in range(rt_dataset.shape[1]):
        new_dataset[:, i] = np.interp(all_freqs, freq_dataset, rt_dataset[:, i])
    return new_dataset
    
def figure_2(rt_num, freq_dataset, rt_dataset, interpolated_dataset):
    plt.figure(figsize=(10, 8))
    plt.semilogx(interpolated_dataset[:,rt_num])
    plt.semilogx(freq_dataset, rt_dataset[:,rt_num], 'o')
    plt.xlim((np.log10(1), 24000))
    plt.savefig(os.path.join("figures", "figure_2.png"))



def main():
    print(os.getcwd())
    path = os.path.join("imports", "two-stage-RT-values.mat")
    rt_dataset = import_rt_dataset(path)
    freq_dataset = dataset_freq()
    figure_1(rt_dataset)
    interpolated_dataset = interpolate_dataset_response(rt_dataset, freq_dataset)
    figure_2(3, freq_dataset, rt_dataset, interpolated_dataset)
    
    
    
    

  
    
if __name__ == "__main__":
    main()
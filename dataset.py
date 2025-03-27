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


def interpolate_dataset_response(freq_dataset, rt_dataset, N=2048):
    new_freqs = np.logspace(np.log10(20), np.log10(20000), N)
    new_dataset = np.zeros((N, rt_dataset.shape[1]))
    for i in range(rt_dataset.shape[1]):
        new_dataset[:, i] = np.interp(new_freqs, freq_dataset, rt_dataset[:, i])
    return new_freqs, new_dataset
    
def figure_2(rt_num, freq_dataset, rt_dataset, interpolated_freqs, interpolated_dataset):
    plt.figure(figsize=(10, 8))
    plt.semilogx(interpolated_freqs, interpolated_dataset[:,rt_num])
    plt.semilogx(freq_dataset, rt_dataset[:,rt_num], 'o')
    # plt.xlim((np.log10(1), 24000))
    plt.savefig(os.path.join("figures", "figure_2.png"))

def figure_3(freq_dataset, rt_dataset):
    '''Box Plot of RT values'''
    plt.figure(figsize=(10, 8))
    plt.boxplot(rt_dataset.T)
    
    x_ticks = [f"{int(freq/1000)}k" if freq > 1000 else str(int(freq)) for freq in freq_dataset]
    x_ticks.insert(0, "")  # Insert an empty string as the first xtick
    plt.xticks(range(len(freq_dataset) + 1), x_ticks, fontsize=8)  # Adjust the range accordingly
    plt.title("Reverberation Time Values for Each Frequency")
    plt.savefig(os.path.join("figures", "figure_3.png"))
    # this should ideally be 1 per octave on the X-axis
    

def figure_4(freq_dataset, rt_dataset):
    '''export the median value of the RT dataset'''
    median_rt = np.median(rt_dataset, axis=1)
    print(median_rt.shape)
    plt.figure(figsize=(10, 8))
    plt.semilogx(freq_dataset, median_rt, 'o', label="Median RT")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Median Dataset T$_{60}$ (s)")
    plt.title("Median RT values for each frequency")
    plt.savefig(os.path.join("figures", "figure_4.png"))
    np.save("median_rt.npy", median_rt)



def main():
    path = os.path.join("imports", "two-stage-RT-values.mat")
    rt_dataset = import_rt_dataset(path)
    freq_dataset = dataset_freq()
    interpolated_freqs, interpolated_dataset = interpolate_dataset_response(freq_dataset, rt_dataset, N=128)
    np.save("interpolated_dataset.npy", interpolated_dataset)
    
    figure_1(rt_dataset)
    figure_2(3, freq_dataset, rt_dataset, interpolated_freqs, interpolated_dataset)
    figure_3(freq_dataset, rt_dataset)
    figure_4(freq_dataset, rt_dataset)

    
    
    
    

  
    
if __name__ == "__main__":
    main()
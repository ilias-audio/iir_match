import scipy.io
import numpy as np


class Dataloader:
  def __init__(self, path_to_raw_dataset, interpolation_size=128):
    self.interpolation_size = interpolation_size
    self.import_rt_dataset(path_to_raw_dataset)
    self.num_of_bands = self.raw_dataset.shape[0]
    self.num_of_rt = self.raw_dataset.shape[1]
    assert self.num_of_bands == 31,   "Dataset should have 31 bands"
    assert self.num_of_rt    == 1000, "Dataset should have 1000 RTs"

    self.dataset_freq()
    assert self.dataset_freqs.ndim == 1, "Dataset frequency should be one-dimensional"

    self.interpolate_dataset(self.interpolation_size)
    assert self.dataset.ndim == 2, "Dataset should be two-dimensional"
    assert self.dataset.shape[0] == self.interpolation_size, "Dataset should have self.interpolation_size frequency bands"
    assert self.dataset.shape[1] == 1000, "Dataset should have 1000 RTs"
    assert self.dataset.shape[0] == self.freqs.shape[0], "Dataset and frequency should have the same number of bands"
    assert self.dataset.shape[1] == self.raw_dataset.shape[1], "Dataset and raw dataset should have the same number of RTs"


  def import_rt_dataset(self, path: str):
    raw_dataset = scipy.io.loadmat(path)
    self.raw_dataset = np.array([raw_dataset["rt_"]]).squeeze()


  def dataset_freq(self):
    '''
    taken directly from twoFilters.m, line 40 on Two_stage_filter repo
    f =  10^3 * (2 .^ ([-17:13]/3))
    '''
    self.dataset_freqs = 10**3 * (2 ** (np.arange(-17, 14) / 3))

  def interpolate_dataset(self, N=2048):
    self.freqs = np.logspace(np.log10(20), np.log10(20000), N)
    self.dataset = np.zeros((N, self.num_of_rt))
    for i in range(self.num_of_rt):
      self.dataset[:, i] = np.interp(self.freqs, self.dataset_freqs, self.raw_dataset[:,i])

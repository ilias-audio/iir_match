import scipy.io
import numpy as np


class Dataset:
  def __init__(self, path_to_rt_dataset, parameters):
    interpolation_size = parameters["INTERPOLATION_SIZE"]
    self.import_rt_dataset(path_to_rt_dataset) # Rows are frequency bands, columns are RTs
    self.interpolate_dataset(interpolation_size) # Create values between 1 and 20kHz, with interpolation_size number of bands
    self.compute_median_rt() # Median RT for each band in the raw dataset and the interpolated dataset

    

  def import_rt_dataset(self, path: str):
    raw_dataset = scipy.io.loadmat(path)
    self._raw_dataset = np.array([raw_dataset["rt_"]]).squeeze()
    self.set_num_of_bands_and_rt(self._raw_dataset)
    self.raw_dataset_freq()

  def interpolate_dataset(self, N=128):
    self.set_interpolation_size(N)
    self._freqs = np.logspace(np.log10(1), np.log10(20000), N)
    self._dataset = np.zeros((N, self.num_of_rt))

    for i in range(self.num_of_rt):
      self._dataset[:, i] = np.interp(self.freqs, self.raw_dataset_freqs, self.raw_dataset[:,i])

    assert self.dataset.ndim == 2, "Dataset should be two-dimensional"
    assert self.dataset.shape[0] == self._interpolation_size, "Dataset should have self._interpolation_size frequency bands"
    assert self.dataset.shape[1] == 1000, "Dataset should have 1000 RTs"
    assert self.dataset.shape[0] == self.freqs.shape[0], "Dataset and frequency should have the same number of bands"
    assert self.dataset.shape[1] == self.raw_dataset.shape[1], "Dataset and raw dataset should have the same number of RTs"

  def compute_median_rt(self):
    self._raw_median_rt = np.array(np.median(self._raw_dataset, axis=1), dtype='f')
    self._median_rt = np.array(np.median(self.dataset, axis=1), dtype='f')
    assert self.raw_median_rt.shape[0] == self.num_of_bands
    assert self.median_rt.shape[0] == self.interpolation_size

  def set_num_of_bands_and_rt(self, raw_dataset):
    assert raw_dataset.ndim == 2, "Raw dataset should be two-dimensional"
    self._raw_dataset = raw_dataset
    self._num_of_bands = raw_dataset.shape[0]
    self._num_of_rt = raw_dataset.shape[1]

  def set_interpolation_size(self, interpolation_size):
    assert interpolation_size > 0, "Interpolation size should be greater than 0"
    assert isinstance(interpolation_size, int), "Interpolation size should be an integer"
    self._interpolation_size = interpolation_size

  def raw_dataset_freq(self):
    '''
    taken directly from twoFilters.m, line 40 on Two_stage_filter repo
    f =  10^3 * (2 .^ ([-17:13]/3))
    '''
    self._raw_dataset_freqs = 10**3 * (2 ** (np.arange(-17, 14) / 3))
    assert self._raw_dataset_freqs.ndim == 1, "Dataset frequency should be one-dimensional"

  @property
  def raw_dataset(self):
    return self._raw_dataset
  
  @property
  def dataset(self):
    return self._dataset

  @property
  def median_rt(self):
    return self._median_rt

  @property
  def raw_median_rt(self):
    return self._raw_median_rt
  
  @property
  def interpolation_size(self):
    return self._interpolation_size
  
  @property
  def num_of_bands(self):
    return self._raw_dataset.shape[0]

  @property
  def num_of_rt(self):
    return self._raw_dataset.shape[1]
  
  @property
  def freqs(self):
    return self._freqs

  @property
  def raw_dataset_freqs(self):
    return self._raw_dataset_freqs
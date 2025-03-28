import os

import Dataloader


if __name__ == "__main__":
  raw_dataset_path = os.path.join("data", "two-stage-RT-values.mat")
  dataloader = Dataloader.Dataloader(raw_dataset_path)
  print(dataloader.raw_dataset.shape)
  print(dataloader.dataset.shape)
  print(dataloader.freqs)  # print(dataloader.num_of_bands)

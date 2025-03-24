import scipy.io
import os
import numpy as np

def import_rt_dataset(path: str):
    raw_dataset = scipy.io.loadmat(path)
    return np.array([raw_dataset["rt_"]]).squeeze()
    


def main():
    print(os.getcwd())
    path = os.path.join("imports", "two-stage-RT-values.mat")
    rt_dataset = import_rt_dataset(path)

    
    
if __name__ == "__main__":
    main()
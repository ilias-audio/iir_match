import matplotlib.pyplot as plt 
import os
import numpy as np
import scipy.signal as signal

class PrintUtilities:
    def __init__(self):
        self.figure_folder = os.path.join(os.curdir, 'figures')
        if not os.path.exists(self.figure_folder):
            os.mkdir(self.figure_folder)

    def SimplePlot(self, x_data, y_data, figure_name: str):
        figure_path = os.path.join(self.figure_folder, figure_name)
        plt.plot(x_data.detach().numpy(), y_data.detach().numpy())
        plt.savefig(figure_path)

    def SemilogxPlot(self, x_data, y_data, figure_name: str):
        figure_path = os.path.join(self.figure_folder, figure_name)
        plt.semilogx(x_data.detach().numpy(), y_data.detach().numpy())
        plt.savefig(figure_path)

    def FrequencyResponse(self, x_data, y_data, figure_name: str):
        figure_path = os.path.join(self.figure_folder, figure_name)
        plt.semilogx(x_data.detach().numpy(), 20 * np.log10(y_data.detach().numpy()))
        plt.savefig(figure_path)

    def PlotIRFreqz(self, ir, figure_name: str, fs = 48000):
        w, h = signal.freqz(ir.detach().numpy().squeeze(), 1, fs = fs)
        figure_path = os.path.join(self.figure_folder, figure_name)
        plt.semilogx(w, 20 * np.log10(abs(h)))
        plt.savefig(figure_path)

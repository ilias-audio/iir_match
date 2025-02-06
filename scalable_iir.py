import torch
import matplotlib.pyplot as plt
import numpy as np
from Filters import *

DELAY_LINES = 1

SAMPLE_RATE = 48000

MIN_DELAY_IN_S = 0.01
MAX_DELAY_IN_S = 0.5
MIN_DELAY_IN_SAMPLES = round(MIN_DELAY_IN_S * SAMPLE_RATE)
MAX_DELAY_IN_SAMPLES = round(MAX_DELAY_IN_S * SAMPLE_RATE)

NUM_OF_FREQS = 16

freqs = torch.logspace(torch.log10(torch.tensor(20.)), torch.log10(torch.tensor(20000.)), NUM_OF_FREQS)


DELAY_LENGTH = torch.randint(MIN_DELAY_IN_SAMPLES, MAX_DELAY_IN_SAMPLES,(DELAY_LINES,1))



RT_TIMES = torch.ones((1,NUM_OF_FREQS))
RT_TIMES[0,8] = 0.1



def convert_rt_to_freq(rt, delay_length, fs):
    return torch.mul((-60.) / (fs *(rt)) , delay_length)



freq_resp = convert_rt_to_freq(RT_TIMES, DELAY_LENGTH, SAMPLE_RATE)

# print(freq_resp.shape)
freq_resp = freq_resp.transpose(0,1)
# print(freq_resp.shape)

dynamic_gain_in_db = (-60 * (DELAY_LENGTH) / (SAMPLE_RATE * (1/(1-RT_TIMES[0,8])* RT_TIMES[0,8])))

offset_gain_in_db = -60 * (DELAY_LENGTH) / ((SAMPLE_RATE))


print("Dynamic Gain= ", dynamic_gain_in_db.T)
print("Offset Gain= ", offset_gain_in_db.T)
print("Resp= ", freq_resp[8])
print("Out by= ", freq_resp[8] - (dynamic_gain_in_db +offset_gain_in_db).T)


offset_gain_linear = torch.pow(10.0, offset_gain_in_db/20.0).squeeze()

plt.clf()
plt.semilogx(freqs, freq_resp)

for i in range(DELAY_LINES):
    Bell_filter = RBJ_Bell(freqs, freqs[8], dynamic_gain_in_db[i], q_factor=4.)

    filter_and_offset_resp = (offset_gain_linear * Bell_filter.response)
    # filter_and_offset_resp = filter_and_offset_resp.transpose(0,1)
    plt.semilogx(freqs, 20. * torch.log10(filter_and_offset_resp))
plt.savefig("freq_resp.png")








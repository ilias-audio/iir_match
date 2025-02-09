'''
generates an arbitrary frequency response for a rt60 and delay length
'''

import numpy as np
import matplotlib.pyplot as plt

NUM_OF_FREQS = 16
NUM_OF_DELAYS = 6
SAMPLE_RATE = 4800

frequencies = np.logspace(np.log10(20.), np.log10(20000), NUM_OF_FREQS)
rt60s = np.linspace(1, 1, NUM_OF_FREQS)
rt60s[8] = 0.5
delay_lengths = np.linspace(0.1, 0.6, NUM_OF_FREQS)

def delay_length_to_samples(delay_length):
    return int(round(delay_length * SAMPLE_RATE))


def convert_rt60_to_freq_responses(freq, rt60s, delay_in_sec, sampling_rate):
    delays_in_samples = np.array([delay_length_to_samples(delay_length) for delay_length in delay_lengths])
    freq_responses = np.zeros((NUM_OF_DELAYS, NUM_OF_FREQS))
    for i in range(NUM_OF_DELAYS):
        freq_responses[i] = (-60 / (sampling_rate * rt60s)) * delays_in_samples[i]

    return freq_responses

res = convert_rt60_to_freq_responses(frequencies, rt60s, delay_lengths, 48000)
for i in range(res.shape[0]):
    plt.semilogx(frequencies, res[i], label=f'Graph {i+1}')
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Response')
plt.legend()
plt.show()




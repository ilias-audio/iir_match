import torch

###############################################################################
# UTILITIES
###############################################################################
def log_normalize(value, min, max):
  value = (torch.log10(value) - torch.log10(min)) / (torch.log10(max) - torch.log10((min)))
  return value

def log_denormalize(value, min, max):
  value = 10 ** (value * torch.log10(max) + (1.0 - value) * torch.log10(min))
  return value

def lin_normalize(value, min, max):
  value = ((value) * (max - min)) + min
  return value

def lin_denormalize(value, min, max):
  value = min + (max - min) * value
  return value


MIN_FREQ = 1.
MAX_FREQ = 19500.

def frequency_denormalize(f):
  min = torch.tensor(MIN_FREQ)
  max = torch.tensor(MAX_FREQ)
  return log_denormalize(f, min, max)

def frequency_normalize(f):
  min = torch.tensor(MIN_FREQ)
  max = torch.tensor(MAX_FREQ)
  return log_normalize(f, min, max)

MAX_GAIN_DB =  250
MIN_GAIN_DB = - 250.

def gain_denormalize(g):
  min = torch.tensor(MIN_GAIN_DB)
  max = torch.tensor(MAX_GAIN_DB)
  return lin_denormalize(g, min, max)

def gain_normalize(g):
  min = torch.tensor(MIN_GAIN_DB)
  max = torch.tensor(MAX_GAIN_DB)
  return lin_normalize(g, min, max)

MIN_Q = 0.01
MAX_Q = 20.

def q_denormalize(q):
  min = torch.tensor(MIN_Q)
  max = torch.tensor(MAX_Q)
  return log_denormalize(q, min, max)

def q_normalize(q):
  min = torch.tensor(MIN_Q)
  max = torch.tensor(MAX_Q)
  return log_normalize(q, min, max)


def convert_proto_gain_to_delay(gamma, delays, fs):
  gain_dB = gamma * (delays / fs)
  return gain_dB.T

def convert_response_to_rt(response_dB, delay, sample_rate):
  rt = (-60 * delay) / (sample_rate * response_dB)
  return rt
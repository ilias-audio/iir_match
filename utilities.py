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
  value = (value - min) / (max - min)
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
  return gain_dB

def convert_response_to_rt(response_dB, delay, sample_rate):
  rt = (-60 * delay) / (sample_rate * response_dB)
  return rt

def check_parameter_bounds(freq, gain, q, name=""):
    """Debug utility to check if parameters are in reasonable ranges"""
    print(f"=== {name} Parameter Check ===")
    print(f"Frequencies: min={freq.min().item():.1f}, max={freq.max().item():.1f}, mean={freq.mean().item():.1f}")
    print(f"Gains: min={gain.min().item():.2f}, max={gain.max().item():.2f}, mean={gain.mean().item():.2f}")
    print(f"Q values: min={q.min().item():.3f}, max={q.max().item():.3f}, mean={q.mean().item():.3f}")
    
    # Check for problematic values
    if freq.min() < 20 or freq.max() > 20000:
        print("WARNING: Frequencies outside audible range!")
    if torch.abs(gain).max() > 24:
        print("WARNING: Extreme gain values detected!")
    if q.min() < 0.1 or q.max() > 10:
        print("WARNING: Extreme Q values detected!")
    print()
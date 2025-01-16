import torch
from tqdm import tqdm
import numpy as np
def lowshelf(f,fc, A, Q):
        W = f / fc
        WW = W*W
        gain_q = torch.sqrt(A) / Q

        num_real = torch.pow((A - WW), 2.)
        num_img  = torch.pow(gain_q * W, 2.)
        num = torch.sqrt(num_real + num_img)

        den_real = torch.pow((1. - A * WW), 2.)
        den_img  = num_img
        den =  torch.sqrt(den_real + den_img)

        return A * (num / den)


def bell_filter(f, f_c, A, Q):
  #A = 10 **(g/40.0)
  num_term1 = (1.0 - (f / f_c) **2 ) ** 2
  num_term2 = ((f / f_c) *  (A / Q)) ** 2
  num = torch.sqrt(num_term1 + num_term2)

  den_term1 = num_term1
  den_term2 = ((f / f_c) / (A * Q)) ** 2
  den = torch.sqrt(den_term1 + den_term2)

  response = num / den

  return response

def evaluate_mag_response(
    f: torch.Tensor,
    f_c_1: torch.Tensor,
    G_1: torch.Tensor,
    Q_1: torch.Tensor,
    f_c_2: torch.Tensor,
    G_2: torch.Tensor,
    Q_2: torch.Tensor,
    f_c_3: torch.Tensor,
    G_3: torch.Tensor,
    Q_3: torch.Tensor
):

  bell_1 = lowshelf(f, f_c_1, G_1, Q_1)
  bell_2 = bell_filter(f, f_c_2, G_2, Q_2)
  bell_3 = bell_filter(f, f_c_3, G_3, Q_3)


  response = bell_1 * bell_2 * bell_3


  return response



def log_normalize(value, min, max):
  value = (torch.log(value) - torch.log(min)) / (torch.log(max) - torch.log((min)))
  return (value)

def log_denormalize(value, min, max):
  value = 10 ** (value * torch.log10(max) + (1.0 - value) * torch.log10(min))
  return value


def lin_normalize(value, min, max):
  value = ((value) * (max - min)) + min
  return value

def lin_denormalize(value, min, max):
  value = min + (max - min) * value
  return value

def frequency_denormalize(f):
  min = torch.tensor(20.0)
  max = torch.tensor(20000.0)

  f = log_denormalize(f, min, max)
  return f

def frequency_normalize(f):
  min = torch.tensor(20.0)
  max = torch.tensor(20000.0)

  f = log_normalize(f, min, max)
  return f

def gain_denormalize(g):
  min = torch.tensor(-12.0)
  max = torch.tensor(12.0)

  real_gain = lin_denormalize(g, min, max)

  g = 10.0 ** (real_gain/40.0)

  return g

def gain_normalize(g):
  min = torch.tensor(-12.0)
  max = torch.tensor(12.0)

  g = lin_normalize(g, min, max)

  return g

def q_denormalize(q):

  min = torch.tensor(0.01)
  max = torch.tensor(4.0)

  return log_denormalize(q, min, max)

def q_normalize(q):

  min = torch.tensor(0.01)
  max = torch.tensor(4.0)

  return log_normalize(q, min, max)

# set the range of frequencies that we evaluate over
f = torch.linspace(20, 20000, 2000)

# target_F1 = frequency_denormalize(torch.rand(1))
# target_G1 = gain_denormalize(torch.rand(1))
# target_Q1 = q_denormalize(torch.rand(1))

# target_F2 = frequency_denormalize(torch.rand(1))
# target_G2 = gain_denormalize(torch.rand(1))
# target_Q2 = q_denormalize(torch.rand(1))

target_F1 = (torch.tensor(1000.0))
target_G1 = (torch.tensor(10.0**(3.0/40)))
target_Q1 = (torch.tensor(1.0))

target_F2 = (torch.tensor(5000.0))
target_G2 = (torch.tensor(10.0**(6.0/40)))
target_Q2 = (torch.tensor(0.5))

target_F3 = (torch.tensor(500.0))
target_G3 = (torch.tensor(10.0**(-2.0/40)))
target_Q3 = (torch.tensor(0.9))

target_response = evaluate_mag_response(f, target_F1, target_G1, target_Q1, 
                                        target_F2, target_G2, target_Q2,
                                        target_F3, target_G3, target_Q3)

print(target_response.max())

n_iters = 10000

parameters = torch.nn.ParameterList()

parameters.append(torch.tensor([0.1, .9, .0]))
parameters.append(torch.tensor([0.9, .9, .0]))
parameters.append(torch.tensor([0.2, .9, .0]))

# F1 = torch.nn.Parameter(torch.tensor(.1))
# G1 = torch.nn.Parameter(torch.tensor(.9))
# Q1 = torch.nn.Parameter(torch.tensor(.0))

# F2 = torch.nn.Parameter(torch.tensor(0.9))
# G2 = torch.nn.Parameter(torch.tensor(0.9))
# Q2 = torch.nn.Parameter(torch.tensor(0.0))

optimizer = torch.optim.Adam(parameters, lr=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iters)

print("TARGET")
print(((target_F1)),
      ((20*np.log10(target_G1))),
      ((target_Q1)),
      ((target_F2)),
      ((20*np.log10(target_G2))),
      ((target_Q2)),
      ((target_F3)),
      ((20*np.log10(target_G3))),
      ((target_Q3)))

pbar = tqdm(range(n_iters))
for n in pbar:
  # reset gradients
  optimizer.zero_grad()

  # evaluate response of current solution
  #pred_response = evaluate_mag_response(f, frequency_denormalize(F1), gain_denormalize(G1), q_denormalise(Q1), frequency_denormalize(F2), gain_denormalize(G2),  q_denormalise(Q2))
  pred_response = evaluate_mag_response(
      f,
      frequency_denormalize(torch.sigmoid(parameters[0][0])),
      gain_denormalize(torch.sigmoid(parameters[0][1])),
      q_denormalize(torch.sigmoid(parameters[0][2])),
      frequency_denormalize(torch.sigmoid(parameters[1][0])),
      gain_denormalize(torch.sigmoid(parameters[1][1])),
      q_denormalize(torch.sigmoid(parameters[1][2])),
      frequency_denormalize(torch.sigmoid(parameters[2][0])),
      gain_denormalize(torch.sigmoid(parameters[2][1])),
      q_denormalize(torch.sigmoid(parameters[2][2]))
    )



  # measure the MSE
  loss = torch.nn.functional.mse_loss((pred_response), (target_response))
  loss.backward()
  optimizer.step()
  scheduler.step()

  pbar.set_description(f"{loss.item():0.4e}")

print("PREDICTION")
print(frequency_denormalize(torch.sigmoid(parameters[0][0])),
      gain_denormalize(torch.sigmoid(parameters[0][1])),
      q_denormalize(torch.sigmoid(parameters[0][2])),
      frequency_denormalize(torch.sigmoid(parameters[1][0])),
      gain_denormalize(torch.sigmoid(parameters[1][1])),
      q_denormalize(torch.sigmoid(parameters[1][2])),
      frequency_denormalize(torch.sigmoid(parameters[2][0])),
      gain_denormalize(torch.sigmoid(parameters[2][1])),
      q_denormalize(torch.sigmoid(parameters[2][2])))
print("ERROR")


import matplotlib.pyplot as plt

plt.semilogx(f, 20 * np.log10(pred_response.detach().numpy()), label="Prediction")

plt.plot(np.array([target_F1.detach().numpy(), target_F2.detach().numpy(), target_F3.detach().numpy()]), np.array([40*np.log10(target_G1.detach().numpy()) , 40*np.log10(target_G2.detach().numpy()), 40*np.log10(target_G3.detach().numpy())]), 'o')


plt.semilogx(f, 20 * np.log10(target_response.detach().numpy()), label="Target")
plt.legend()
plt.title("Frequency response matching using stochastic gradient descent")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.savefig("./match.png")
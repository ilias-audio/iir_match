import torch
import torchaudio

class RBJ_LowShelf:
    
    def __init__(self, current_freq, cut_off, gain, q_factor=0.7071, sample_rate = 48000):
        self.cut_off = cut_off
        self.current_freq = current_freq 
        self.gain = gain
        self.q_factor = q_factor
        self.sample_rate = sample_rate
        self.set_linear_gain(gain)
        self.compute_response()
        # self.sos = torch.zeros((1,6))
        self.compute_sos()

    def set_linear_gain(self, gain):
        self.A = torch.pow(10.0,(gain/40.0))

    def compute_response(self):
        W = self.current_freq.unsqueeze(-1) / self.cut_off.unsqueeze(0)
        WW = W*W
        gain_q = torch.sqrt(self.A) / self.q_factor

        num_real = torch.pow((self.A - WW), 2.)
        num_img  = torch.pow(gain_q * W, 2.)
        num = torch.sqrt(num_real + num_img)

        den_real = torch.pow((1. - self.A * WW), 2.)
        den_img  = num_img
        den =  torch.sqrt(den_real + den_img)

        self.response = self.A * (num / den)

    def compute_sos(self):
        w0 = 2.0 * torch.pi * (self.cut_off/self.sample_rate)
        alpha = torch.sin(w0) / (2.0 * self.q_factor)
        A = self.A
        
        b0 =     A*((A+1.) - (A-1.)*torch.cos(w0) + 2*torch.sqrt(A)*alpha )
        b1 =  2.*A*((A-1.) - (A+1.)*torch.cos(w0)                         )
        b2 =     A*((A+1.) - (A-1.)*torch.cos(w0) - 2*torch.sqrt(A)*alpha )
        a0 =        (A+1.) + (A-1.)*torch.cos(w0) + 2*torch.sqrt(A)*alpha
        a1 =   -2.*((A-1.) + (A+1.)*torch.cos(w0)                         )
        a2 =        (A+1.) + (A-1.)*torch.cos(w0) - 2*torch.sqrt(A)*alpha

        # Numerator
        b0 /= a0
        b1 /= a0
        b2 /= a0
        # Denumerator
        
        a1 /= a0
        a2 /= a0
        
        a0 = torch.ones_like(b0)
        
        self.sos = torch.stack([b0, b1, b2, a0, a1, a2], dim=1)
        omega = torch.linspace(0, torch.pi, len(self.current_freq), dtype=torch.float32)  # [0, 2*pi fc/fs]
        e_jw = torch.exp(-1j * omega).unsqueeze(-1)  # e^{-jω}
        e_jw2 = e_jw ** 2
        num = b0 + e_jw * b1 + e_jw2 * b2
        den = 1 + e_jw * a1 + e_jw2 * a2
        self.sos_response = (num / den)
        
        

class RBJ_HighShelf:
    
    def __init__(self, current_freq, cut_off, gain, q_factor=0.7071, sample_rate = 48000):
        self.cut_off = cut_off
        self.current_freq = current_freq 
        self.gain = gain
        self.q_factor = q_factor
        self.sample_rate = sample_rate
        self.set_linear_gain(gain)
        self.compute_response()
        self.compute_sos()


    def set_linear_gain(self, gain):
        self.A = torch.pow(10.0,(gain/40.0))

    def compute_response(self):
        W = self.current_freq.unsqueeze(-1) / self.cut_off.unsqueeze(0)
        WW = W*W
        gain_q = torch.sqrt(self.A) / self.q_factor

        num_real = torch.pow((1. - self.A * WW), 2.)
        num_img  = torch.pow(gain_q * W, 2.)
        num = torch.sqrt(num_real + num_img)

        den_real = torch.pow((self.A - WW), 2.)
        den_img  = num_img
        den =  torch.sqrt(den_real + den_img)

        self.response = self.A * (num / den)
        
        
    def compute_sos(self):
        w0 = 2.0 * torch.pi * (self.cut_off/self.sample_rate)
        alpha = torch.sin(w0) / (2.0 * self.q_factor)
        A = self.A
        
        b0 =    A*( (A+1.) + (A-1.)*torch.cos(w0) + 2*torch.sqrt(A)*alpha )
        b1 = -2.*A*( (A-1.) + (A+1.)*torch.cos(w0)                   )
        b2 =    A*( (A+1.) + (A-1.)*torch.cos(w0) - 2*torch.sqrt(A)*alpha )
        a0 =        (A+1.) - (A-1.)*torch.cos(w0) + 2*torch.sqrt(A)*alpha
        a1 =    2.*( (A-1.) - (A+1.)*torch.cos(w0)                   )
        a2 =        (A+1.) - (A-1.)*torch.cos(w0) - 2*torch.sqrt(A)*alpha


        # Numerator
        b0 /= a0
        b1 /= a0
        b2 /= a0
        # Denumerator
        
        a1 /= a0
        a2 /= a0
        
        a0 = torch.ones_like(b0)
        
        self.sos = torch.stack([b0, b1, b2, a0, a1, a2], dim=1)
        omega = torch.linspace(0, torch.pi, len(self.current_freq), dtype=torch.float32)  # [0, 2*pi fc/fs]
        e_jw = torch.exp(-1j * omega).unsqueeze(-1)  # e^{-jω}
        e_jw2 = e_jw ** 2
        num = b0 + e_jw * b1 + e_jw2 * b2
        den = 1 + e_jw * a1 + e_jw2 * a2
        self.sos_response = (num / den)


class RBJ_Bell:
    def __init__(self, current_freq, cut_off, gain, q_factor=0.7071, sample_rate = 48000):
        self.cut_off = cut_off
        self.current_freq = current_freq 
        self.gain = gain
        self.q_factor = q_factor
        self.sample_rate = sample_rate
        self.set_linear_gain(gain)
        self.compute_response()
        self.compute_sos()


    def set_linear_gain(self, gain):
        self.A = torch.pow(10.0,(gain/40.0))

    def compute_response(self):
        W = self.current_freq.unsqueeze(-1) / self.cut_off.unsqueeze(0)
        WW = W*W
        gain_q = self.A / self.q_factor
        inv_gainq = 1. / (self.A * self.q_factor)

        num_real = torch.pow((1. - WW), 2.)
        num_img  = torch.pow(gain_q * W, 2.)
        num = torch.sqrt(num_real + num_img)

        den_real = torch.pow((1. - WW), 2.)
        den_img  = torch.pow(inv_gainq * W, 2.)
        den =  torch.sqrt(den_real + den_img)

        self.response = (num / den)
        
    def compute_sos(self):
        w0 = 2. * torch.pi * (self.cut_off/self.sample_rate)
        alpha = torch.sin(w0) / (2.0 * self.q_factor)
        A = self.A        
        b0 =   1. + alpha*A
        b1 =  -2.*torch.cos(w0)
        b2 =   1. - alpha*A
        a0 =   1. + alpha/A
        a1 =  -2.*torch.cos(w0)
        a2 =   1. - alpha/A


        # Numerator
        b0 /= a0
        b1 /= a0
        b2 /= a0
        # Denumerator
        a1 /= a0
        a2 /= a0
        
        a0 = torch.ones_like(b0)
        
        
        self.sos = torch.stack([b0, b1, b2, a0, a1, a2], dim=1)
        omega = torch.linspace(0, torch.pi, len(self.current_freq), dtype=torch.float32)  # [0, 2*pi fc/fs]
        e_jw = torch.exp(-1j * omega).unsqueeze(-1)  # e^{-jω}
        e_jw2 = e_jw ** 2
        num = b0 + e_jw * b1 + e_jw2 * b2
        den = 1 + e_jw * a1 + e_jw2 * a2
        self.sos_response = (num / den)



def evaluate_mag_response(
  x: torch.Tensor,     # Frequency vector
  F: torch.Tensor,     # Center frequencies (NUM_OF_DELAYS x NUM_OF_BANDS)
  G: torch.Tensor,     # Gain values (NUM_OF_DELAYS x NUM_OF_BANDS)
  Q: torch.Tensor      # Q values (NUM_OF_DELAYS x NUM_OF_BANDS)
):
  assert F.shape == G.shape == Q.shape, "All parameter arrays must have the same shape"
  NUM_OF_BANDS, NUM_OF_DELAYS = F.shape
  
  # Initialize response tensor
  response = torch.ones((len(x), NUM_OF_DELAYS), device=x.device)
  
  # Compute responses for each band
  low_shelf_responses = RBJ_LowShelf(x, F[0, :], G[0, :], Q[0, :]).response
  bell_responses = torch.stack([RBJ_Bell(x, F[i, :], G[i, :], Q[i, :]).response for i in range(1, NUM_OF_BANDS-1)], dim=1)
  high_shelf_responses = RBJ_HighShelf(x, F[-1, :], G[-1, :], Q[-1, :]).response
  
  # Combine responses
  response *= low_shelf_responses
  response *= torch.prod(bell_responses, dim=1)
  response *= high_shelf_responses

  return response

def evaluate_sos_response(
  x: torch.Tensor,     # Frequency vector
  F: torch.Tensor,     # Center frequencies (NUM_OF_DELAYS x NUM_OF_BANDS)
  G: torch.Tensor,     # Gain values (NUM_OF_DELAYS x NUM_OF_BANDS)
  Q: torch.Tensor      # Q values (NUM_OF_DELAYS x NUM_OF_BANDS)
):
  assert F.shape == G.shape == Q.shape, "All parameter arrays must have the same shape"
  NUM_OF_BANDS, NUM_OF_DELAYS = F.shape
  
  # Initialize response tensor
  response = torch.ones((len(x), NUM_OF_DELAYS), device=x.device, dtype=torch.cfloat)
  
  # Compute responses for each band
  low_shelf_responses = RBJ_LowShelf(x, F[0, :], G[0, :], Q[0, :]).sos_response
  bell_responses = torch.stack([RBJ_Bell(x, F[i, :], G[i, :], Q[i, :]).sos_response for i in range(1, NUM_OF_BANDS-1)], dim=1)
  high_shelf_responses = RBJ_HighShelf(x, F[-1, :], G[-1, :], Q[-1, :]).sos_response
  
  # Combine responses
  response *= low_shelf_responses
  response *= torch.prod(bell_responses, dim=1)
  response *= high_shelf_responses

  return response
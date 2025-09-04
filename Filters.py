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
        # self.impulse_response(self.sos[:,0:3], self.sos[:,3:6], int(24e3))

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
        # omega = torch.linspace(0, torch.pi, len(self.current_freq), dtype=torch.float32)  # [0, 2*pi fc/fs]
        # e_jw = torch.exp(-1j * omega).unsqueeze(-1)  # e^{-jω}
        # e_jw2 = e_jw ** 2
        # num = b0 + e_jw * b1 + e_jw2 * b2
        # den = 1 + e_jw * a1 + e_jw2 * a2
        # self.sos_response = (num / den)

    def impulse_response(self, B, A, n_samples):
        B = torch.tensor(B, dtype=torch.float32)  # shape: (B, M)
        A = torch.tensor(A, dtype=torch.float32)  # shape: (B, N)
        batch_size, M = B.shape
        _, N = A.shape

        assert torch.allclose(A[:, 0], torch.ones_like(A[:, 0])), "A[:, 0] must be 1"

        x = torch.zeros((batch_size, n_samples))
        x[:, 0] = 1.0  # unit impulse for each batch
        y = torch.zeros((batch_size, n_samples))

        for n in range(n_samples):
            # feedforward
            for i in range(M):
                if n - i >= 0:
                    y[:, n] += B[:, i] * x[:, n - i]
            # feedback
            for j in range(1, N):
                if n - j >= 0:
                    y[:, n] -= A[:, j] * y[:, n - j]

        self.impulse_response = y        
        

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
        # self.impulse_response(self.sos[:,0:3], self.sos[:,3:6], int(24e3))


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
        # omega = torch.linspace(0, torch.pi, len(self.current_freq), dtype=torch.float32)  # [0, 2*pi fc/fs]
        # e_jw = torch.exp(-1j * omega).unsqueeze(-1)  # e^{-jω}
        # e_jw2 = e_jw ** 2
        # num = b0 + e_jw * b1 + e_jw2 * b2
        # den = 1 + e_jw * a1 + e_jw2 * a2
        # self.sos_response = (num / den)

    def impulse_response(self, B, A, n_samples):
        B = torch.tensor(B, dtype=torch.float32)  # shape: (B, M)
        A = torch.tensor(A, dtype=torch.float32)  # shape: (B, N)
        batch_size, M = B.shape
        _, N = A.shape

        assert torch.allclose(A[:, 0], torch.ones_like(A[:, 0])), "A[:, 0] must be 1"

        x = torch.zeros((batch_size, n_samples))
        x[:, 0] = 1.0  # unit impulse for each batch
        y = torch.zeros((batch_size, n_samples))

        for n in range(n_samples):
            # feedforward
            for i in range(M):
                if n - i >= 0:
                    y[:, n] += B[:, i] * x[:, n - i]
            # feedback
            for j in range(1, N):
                if n - j >= 0:
                    y[:, n] -= A[:, j] * y[:, n - j]

        self.impulse_response = y

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
        # self.impulse_response(self.sos[:,0:3], self.sos[:,3:6], int(24e3))


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
        # omega = torch.linspace(0, torch.pi, len(self.current_freq), dtype=torch.float32)  # [0, 2*pi fc/fs]
        # e_jw = torch.exp(-1j * omega).unsqueeze(-1)  # e^{-jω}
        # e_jw2 = e_jw ** 2
        # num = b0 + e_jw * b1 + e_jw2 * b2
        # den = 1 + e_jw * a1 + e_jw2 * a2
        # self.sos_response = (num / den)

    import torch

    def impulse_response(self, B, A, n_samples):
        B = torch.tensor(B, dtype=torch.float32)  # shape: (B, M)
        A = torch.tensor(A, dtype=torch.float32)  # shape: (B, N)
        batch_size, M = B.shape
        _, N = A.shape

        assert torch.allclose(A[:, 0], torch.ones_like(A[:, 0])), "A[:, 0] must be 1"

        x = torch.zeros((batch_size, n_samples))
        x[:, 0] = 1.0  # unit impulse for each batch
        y = torch.zeros((batch_size, n_samples))

        for n in range(n_samples):
            # feedforward
            for i in range(M):
                if n - i >= 0:
                    y[:, n] += B[:, i] * x[:, n - i]
            # feedback
            for j in range(1, N):
                if n - j >= 0:
                    y[:, n] -= A[:, j] * y[:, n - j]

        self.impulse_response = y




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

#   ir = torch.zeros(( NUM_OF_DELAYS, int(24e3)), device=x.device, dtype=torch.float)
  
  # Compute responses for each band
  low_shelf_responses = RBJ_LowShelf(x, F[0, :], G[0, :], Q[0, :]).impulse_response
  high_shelf_responses = RBJ_HighShelf(x, F[-1, :], G[-1, :], Q[-1, :]).impulse_response
  
  bell_responses = torch.stack([RBJ_Bell(x, F[i, :], G[i, :], Q[i, :]).impulse_response for i in range(1, NUM_OF_BANDS-1)], dim=1)
  B, F, L = bell_responses.shape
  out_len = (L - 1) * F + 1
  out = torch.zeros(B, 1, out_len)

  for b in range(B):
    y = bell_responses[b, 0].view(1, 1, -1)
    for f in range(1, F):
      h = bell_responses[b, f].view(1, 1, -1)
      y = torch.nn.functional.conv1d(y, h)
    out[b, 0, :y.shape[-1]] = y
  
  # Combine responses
  ir = low_shelf_responses.unsqueeze(0)
  ir = torch.nn.functional.conv1d(ir, out, groups=ir.shape[0])
  ir =  torch.nn.functional.conv1d(ir, high_shelf_responses.unsqueeze(1),groups=ir.shape[0])
  return ir, response
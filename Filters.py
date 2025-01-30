import torch

class RBJ_LowShelf:
    
    def __init__(self, current_freq, cut_off, gain, q_factor=0.7071):
        self.cut_off = cut_off
        self.current_freq = current_freq 
        self.gain = gain
        self.q_factor = q_factor
        self.set_linear_gain(gain)
        self.compute_response()


    def set_linear_gain(self, gain):
        self.A = torch.pow(10.0,(gain/40.0))

    def compute_response(self):
        W = self.current_freq / self.cut_off
        WW = W*W
        gain_q = torch.sqrt(self.A) / self.q_factor

        num_real = torch.pow((self.A - WW), 2.)
        num_img  = torch.pow(gain_q * W, 2.)
        num = torch.sqrt(num_real + num_img)

        den_real = torch.pow((1. - self.A * WW), 2.)
        den_img  = num_img
        den =  torch.sqrt(den_real + den_img)

        self.response = self.A * (num / den)


class RBJ_HighShelf:
    
    def __init__(self, current_freq, cut_off, gain, q_factor=0.7071):
        self.cut_off = cut_off
        self.current_freq = current_freq 
        self.gain = gain
        self.q_factor = q_factor
        self.set_linear_gain(gain)
        self.compute_response()


    def set_linear_gain(self, gain):
        self.A = torch.pow(10.0,(gain/40.0))

    def compute_response(self):
        W = self.current_freq / self.cut_off
        WW = W*W
        gain_q = torch.sqrt(self.A) / self.q_factor

        num_real = torch.pow((1. - self.A * WW), 2.)
        num_img  = torch.pow(gain_q * W, 2.)
        num = torch.sqrt(num_real + num_img)

        den_real = torch.pow((self.A - WW), 2.)
        den_img  = num_img
        den =  torch.sqrt(den_real + den_img)

        self.response = self.A * (num / den)


class RBJ_Bell:
    def __init__(self, current_freq, cut_off, gain, q_factor=0.7071):
        self.cut_off = cut_off
        self.current_freq = current_freq 
        self.gain = gain
        self.q_factor = q_factor
        self.set_linear_gain(gain)
        self.compute_response()


    def set_linear_gain(self, gain):
        self.A = torch.pow(10.0,(gain/40.0))

    def compute_response(self):
        W = self.current_freq / self.cut_off
        WW = W*W
        gain_q = torch.sqrt(self.A) / self.q_factor
        inv_gainq = 1. / (torch.sqrt(self.A) * self.q_factor)

        num_real = torch.pow((1. - WW), 2.)
        num_img  = torch.pow(gain_q * W, 2.)
        num = torch.sqrt(num_real + num_img)

        den_real = torch.pow((1. - WW), 2.)
        den_img  = torch.pow(inv_gainq * W, 2.)
        den =  torch.sqrt(den_real + den_img)

        self.response = (num / den)
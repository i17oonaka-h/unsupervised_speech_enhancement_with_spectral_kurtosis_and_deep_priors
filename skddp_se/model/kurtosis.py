import torch
import torch.nn as nn

def gamma_kurtosis(Px, eps=1e-8):
    gamma = torch.log(torch.mean(Px, dim=(1,2,3))+eps) - torch.mean(torch.log(Px+eps), dim=(1,2,3))
    eta = (3 - gamma + torch.sqrt((gamma - 3)**2 + 24 * gamma)) / (12 * gamma+eps)
    kurtosis = (eta + 2) * (eta + 3) / (eta * (eta + 1)+eps)
    return kurtosis

class KurtosisLoss(nn.Module):
    def __init__(self, weight, kernel, shift, target, DEVICE="cpu", type_="Power", eps=1e-8, optim_type="increasing", inverse=False):
        super().__init__()
        self.weight = weight
        self.optim_type = optim_type
        self.kurtosis_calculator = SegKurtosis(
            kernel=kernel,
            shift=shift,
            DEVICE=DEVICE,
            type_=type_,
            eps=eps
        )
        self.target_kurtosis = self.kurtosis_calculator(target)
        if inverse:
            self.target_kurtosis = torch.max(self.target_kurtosis) - self.target_kurtosis + torch.min(self.target_kurtosis)

    def forward(self, Px):
        if self.target_kurtosis.device != Px.device:
            self.target_kurtosis = self.target_kurtosis.to(Px.device)
        pred_kurtosis = self.kurtosis_calculator(Px)
        if self.optim_type == "increasing":
            return torch.mean(
                self.weight * torch.mean(
                    (pred_kurtosis/self.target_kurtosis)**2 * (-1)
                )
            )
        elif self.optim_type == "decreasing":
            return torch.mean(
                self.weight * torch.mean(
                    (pred_kurtosis/self.target_kurtosis)**2
                )
            )
        else:
            raise NotImplementedError

    
    

class SegKurtosis(nn.Module):
    def __init__(self, kernel, shift, DEVICE="cpu", type_="Power", eps=1e-8):
        """
        Args:
            kernel (tuple): (frequency_kernel_size, time_kernel_size). -1 means full.
            shift (tuple): (frequency_stride, time_stride). -1 means full.
            DEVICE (str):
            type (str): "Power" or "others"
        
        Note:
            When type is "Power", the input is power spectrogram. 
            kurtosis calculated by modeling signal as Gaussian distribution [1].
            When type is "others", the input is spectrogram.
            kurtosis calculated by sample mean. 
            (
                This one has not been tested in our work, 
                so it is unknown whether it will work well with complex spec., etc.
            )
            kurtosis = (eta + 2) * (eta + 3) / (eta * (eta + 1)) // [1]
            kurtosis = mu_4 / (mu_2**2) // on "others"
        """
        self.type_ = type_
        self.eps = eps
        super().__init__()
        if kernel[0] == -1: # -1 means full
            self.f_meanflag = True
            kernel = (1, kernel[1])
            shift = (1, shift[1])
        else:
            self.f_meanflag = False
        if kernel[1] == -1: # -1 means full
            self.t_meanflag = True
            kernel = (kernel[0], 1)
            shift = (shift[0], 1)
        else:
            self.t_meanflag = False
        self.calculator = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel,
            stride=shift,
            padding_mode="zeros",
            bias=False
        ).float().to(DEVICE) # The division process was implemented in Conv2d.
        torch.nn.init.ones_(self.calculator.weight)
        for param in self.calculator.parameters():
            param.requires_grad = False
        self.calculator.eval()

    def _gamma(self, P_Input_):
        if self.f_meanflag and self.t_meanflag: # kurtosis of whole spectrogram
            seg_meanP = torch.mean(P_Input_, dim=(2,3), keepdim=True)
            A = torch.log(seg_meanP+self.eps)
            B = torch.mean(torch.log(P_Input_+self.eps), dim=(2,3), keepdim=True)
        elif (not self.f_meanflag) and (not self.t_meanflag): # kurtosis of each segment
            seg_sumP = self.calculator(P_Input_)
            seg_num = self.calculator(torch.ones_like(P_Input_))
            seg_meanP = seg_sumP / seg_num
            A = torch.log(seg_meanP+self.eps)
            B = self.calculator(torch.log(P_Input_+self.eps)) / seg_num + self.eps
        else: # kurtosis of each segment, but mean of the other axis
            dim_ = 2 if self.f_meanflag else 3 # fflag: dim=2, tflag: dim=3
            seg_sumP = self.calculator(torch.sum(P_Input_, dim=dim_, keepdim=True))
            seg_num = self.calculator(torch.sum(torch.ones_like(P_Input_), dim=dim_, keepdim=True))
            seg_meanP = seg_sumP / seg_num
            A = torch.log(seg_meanP+self.eps)
            B = self.calculator(torch.sum(torch.log(P_Input_+self.eps), dim=dim_, keepdim=True)) / seg_num
        Gamma = A - B
        Eta = (3 - Gamma + torch.sqrt((Gamma - 3)**2 + 24 * Gamma)) / (12 * Gamma+self.eps)
        Kurtosis = (Eta + 2) * (Eta + 3) / (Eta * (Eta + 1)+self.eps)
        return Kurtosis

    def _sampleby(self, P_Input_):
        if self.f_meanflag and self.t_meanflag:
            seg_meanP2 = torch.mean(P_Input_**2, dim=(1,2,3), keepdim=True)
            seg_meanP4 = torch.mean(P_Input_**4, dim=(1,2,3), keepdim=True)
        elif (not self.f_meanflag) and (not self.t_meanflag):
            seg_sumP2 = self.calculator(P_Input_[:,:1]**2) + self.calculator(P_Input_[:,1:]**2)
            seg_sumP4 = self.calculator(P_Input_[:,:1]**4) + self.calculator(P_Input_[:,1:]**4)
            seg_num = self.calculator(torch.ones_like(P_Input_[:,:1])) + self.calculator(torch.ones_like(P_Input_[:,1:]))
            seg_meanP2 = seg_sumP2 / seg_num
            seg_meanP4 = seg_sumP4 / seg_num
        else:
            dim_ = 2 if self.f_meanflag else 3
            seg_sumP2 = self.calculator(torch.sum(P_Input_**2, dim=dim_))
            seg_sumP4 = self.calculator(torch.sum(P_Input_**4, dim=dim_))
            seg_num = self.calculator(torch.sum(torch.ones_like(P_Input_), dim=dim_))
            seg_meanP2 = seg_sumP2 / seg_num
            seg_meanP4 = seg_sumP4 / seg_num
        mu_2 = seg_meanP2
        mu_4 = seg_meanP4
        Kurtosis = mu_4 / (mu_2**2+self.eps)
        return Kurtosis

    def forward(self, Input_):
        """
        Args:
            Input_ (torch.Tensor): Input power spectrogram (B, 1, F, T)
        Returns:
            Kurtosis (B, F', T')
        """
        if self.type_ == "Power":
            return self._gamma(Input_)
        elif self.type_ == "others":
            return self._sampleby(Input_)
        else:
            raise NotImplementedError
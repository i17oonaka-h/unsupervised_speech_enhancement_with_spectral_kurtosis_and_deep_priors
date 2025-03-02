import torch 
import torch.nn as nn
from skddp_se.utils.config import ModelConfig

class UnetModule(nn.Module):
    def __init__(self, feature_channels=1, base_hidden_channels=35):
        super(UnetModule, self).__init__()
        self.base_hidden_channels = base_hidden_channels
        self.down0 = self._ConvBlock(feature_channels, base_hidden_channels)
        self.down1 = self._ConvBlock(base_hidden_channels, base_hidden_channels*2)
        self.down2 = self._ConvBlock(base_hidden_channels*2, base_hidden_channels*2)
        self.up2 = self._ConvBlock(base_hidden_channels*4, base_hidden_channels)
        self.up1 = self._ConvBlock(base_hidden_channels*2, base_hidden_channels)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.proj = nn.Conv2d(base_hidden_channels, feature_channels, 1)
    
    def forward(self, x):
        # padding
        divisible = 8
        f_padlen = (divisible - x.size(2) % divisible) % divisible
        t_padlen = (divisible - x.size(3) % divisible) % divisible
        x = nn.functional.pad(x, (0, t_padlen, 0, f_padlen))
        # forward
        x0 = self.down0(x)
        x1 = self.down1(self.maxpool(x0))
        x2 = self.down2(self.maxpool(x1))
        x = self.up2(torch.cat([self.upsample(x2), x1], dim=1))
        x = self.up1(torch.cat([self.upsample(x), x0], dim=1))
        x = self.proj(x)
        # remove padding
        x = x[:,:,:-f_padlen,:-t_padlen]
        return x

    class _ConvBlock(nn.Module):
        """ ConvBlock architecture.
        This class implements the ConvBlock architecture.
        model follows under settings:
            ConvBlock(inc,outc): Conv2d(inc,outc)->InstanceNorm->LeakyReLU->Conv2d(outc,outc)->InstanceNorm->LeakyReLU
        """
        def __init__(self, inc, outc):
            """ Initializes ConvBlock.
            Args:
                inc: int, number of input channels
                outc: int, number of output channels
            """
            super(UnetModule._ConvBlock, self).__init__()
            self.conv1 = nn.Conv2d(inc, outc, 3, padding=1)
            self.norm1 = nn.InstanceNorm2d(outc)
            self.relu1 = nn.LeakyReLU(0.2)
            self.conv2 = nn.Conv2d(outc, outc, 3, padding=1)
            self.norm2 = nn.InstanceNorm2d(outc)
            self.relu2 = nn.LeakyReLU(0.2)

        def forward(self, x):
            """ Forward method.
            Args:
                x: torch.Tensor, input tensor (B,inC,F,T)
            Returns:
                torch.Tensor, output tensor (B,outC,F,T)
            """
            x = self.relu1(self.norm1(self.conv1(x)))
            x = self.relu2(self.norm2(self.conv2(x)))
            return x

class SimpleTwinUnet(nn.Module):
    def __init__(
        self, 
        model_config: ModelConfig
    ):
        super(SimpleTwinUnet, self).__init__()
        self.input_features = {
            "clean": model_config.clean_net.input_feature,
            "noise": model_config.noise_net.input_feature
        }
        self.softplus_betas = {
            "clean": model_config.clean_net.softplus_beta,
            "noise": model_config.noise_net.softplus_beta
        }
        self.clean_net = UnetModule(feature_channels=1, base_hidden_channels=35)
        self.mixing_type = model_config.mixing_type
        self.noise_net = UnetModule(feature_channels=1, base_hidden_channels=35)
        self.clean_output_layer = nn.Softplus(beta=self.softplus_betas["clean"])
        self.noise_output_layer = nn.Softplus(beta=self.softplus_betas["noise"])
    
    def get_input_features(self, target):
        # clean
        if self.input_features["clean"] == "uniform":
            featurec = torch.rand_like(target, device=target.device)
        elif self.input_features["clean"] == "uniform-TFcoherent":
            tmpf = torch.rand_like(target[:,:,0:1], device=target.device)
            tmpt = torch.rand_like(target[:,:,:,0:1], device=target.device)
            featurec = (1/2)*(tmpf+tmpt)
        elif self.input_features["clean"] == "meshgrid":
            featurec = torch.meshgrid(torch.arange(0, target.size(2), device=target.device))
            featurec = torch.flip(featurec[0], [0])
            featurec = torch.repeat_interleave(featurec.unsqueeze(1), target.size(3), dim=1)
            featurec = featurec.unsqueeze(0).unsqueeze(0)
            featurec = featurec / torch.max(featurec)
            featurec = 0.9*featurec + 0.1*torch.rand_like(featurec, device=target.device)
        else:
            raise ValueError(f"Invalid input feature: {self.input_features['clean']}")
        # noise
        target = target[:1] # noise batch size is 1
        if self.input_features["noise"] == "uniform":
            featuren = torch.rand_like(target, device=target.device)
        elif self.input_features["noise"] == "uniform-TFcoherent":
            tmpf = torch.rand_like(target[:,:,0:1], device=target.device)
            tmpt = torch.rand_like(target[:,:,:,0:1], device=target.device)
            featuren = (1/2)*(tmpf+tmpt)
        elif self.input_features["noise"] == "meshgrid":
            featuren = torch.meshgrid(torch.arange(0, target.size(2), device=target.device))
            featuren = torch.flip(featuren[0], [0])
            featuren = torch.repeat_interleave(featuren.unsqueeze(1), target.size(3), dim=1)
            featuren = featuren.unsqueeze(0).unsqueeze(0)
            featuren = featuren / torch.max(featuren)
            featuren = 0.9*featuren + 0.1*torch.rand_like(featuren, device=target.device)
        else:
            raise ValueError(f"Invalid input feature: {self.input_features['noise']}")
        return featurec, featuren

    def _clean_mixing(self, clean):
        if self.mixing_type == "mean":
            return torch.mean(clean, dim=0, keepdim=True)
        elif self.mixing_type == "median":
            return torch.median(clean, dim=0, keepdim=True)
        elif self.mixing_type == "min":
            return torch.min(clean, dim=0, keepdim=True).values
        else:
            raise NotImplementedError(f"Invalid mixing type: {self.mixing_type}")

    def forward(self, featurec, featuren):
        clean = self.clean_net(featurec)
        noise = self.noise_net(featuren)
        clean = self.clean_output_layer(clean)
        noise = self.noise_output_layer(noise)
        return clean, self._clean_mixing(clean), noise
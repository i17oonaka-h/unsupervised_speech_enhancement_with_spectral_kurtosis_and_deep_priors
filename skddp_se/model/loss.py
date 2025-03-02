import torch
from skddp_se.utils.config import LossConfig
from skddp_se.model.kurtosis import KurtosisLoss

class skddpLoss(torch.nn.Module):
    def __init__(self, config: LossConfig, target: torch.Tensor):
        super().__init__()
        self.config = config
        self.reconstruction_loss = torch.nn.L1Loss() if config.reconstruction_loss_type in ["MAE"] \
            else torch.nn.MSELoss()
        self.clean_segkurtosis_increasing = KurtosisLoss(
            weight=config.clean_segkurtosis_increasing.weight,
            kernel=config.clean_segkurtosis_increasing.kernel_size,
            shift=config.clean_segkurtosis_increasing.stride,
            target=target.to("cpu"),
            type_="Power",
            eps=1e-8,
            optim_type="increasing",
            inverse=True
        )
        self.noise_segkurtosis_decreasing = KurtosisLoss(
            weight=config.noise_segkurtosis_decreasing.weight,
            kernel=config.noise_segkurtosis_decreasing.kernel_size,
            shift=config.noise_segkurtosis_decreasing.stride,
            target=target.to("cpu"),
            type_="Power",
            eps=1e-8,
            optim_type="decreasing",
            inverse=True
        )
        self.clean_refinement_decreasing = KurtosisLoss(
            weight=config.clean_refinement_decreasing.weight,
            kernel=config.clean_refinement_decreasing.kernel_size,
            shift=config.clean_refinement_decreasing.stride,
            target=target.to("cpu"),
            type_="Power",
            eps=1e-8,
            optim_type="decreasing",
            inverse=False
        )
        self.clean_refinement_increasing = KurtosisLoss(
            weight=config.clean_refinement_increasing.weight,
            kernel=config.clean_refinement_increasing.kernel_size,
            shift=config.clean_refinement_increasing.stride,
            target=target.to("cpu"),
            type_="Power",
            eps=1e-8,
            optim_type="increasing",
            inverse=True
        )
    
    def forward(self, pred_clean, pred_clean_mixed, pred_noise, target_noisy):
        loss_logits = {}
        loss = self.reconstruction_loss(pred_clean+pred_noise, target_noisy)
        loss_logits["Lrec"] = loss.item()
        tmp = self.clean_segkurtosis_increasing(pred_clean)
        loss += tmp
        loss_logits["L1(S)"] = tmp.item()
        tmp = self.noise_segkurtosis_decreasing(pred_noise)
        loss += tmp
        loss_logits["L(N)"] = tmp.item()
        tmp = self.clean_refinement_decreasing(pred_clean_mixed)
        loss += tmp
        loss_logits["L2(S)_term1"] = tmp.item()
        tmp = self.clean_refinement_increasing(pred_clean_mixed)
        loss += tmp
        loss_logits["L2(S)_term2"] = tmp.item()
        return loss, loss_logits
    


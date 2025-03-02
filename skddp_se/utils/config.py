import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Union, Tuple

def set_seed(seed: int) -> None:
    import random
    import torch
    import numpy as np
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    return 

@dataclass
class LoggingConfig:
    log_dir: Path
    log_freq: int
    save_intermediate_steps: List[int]

@dataclass
class PreprocessConfig:
    dump_dir: Path
    sample_rate: int
    noise_list: List[Path]
    clean_list: List[Path]
    clean_samples: Union[int, None]
    SNR_list: List[float]
    clip_seconds: Union[float, None]

@dataclass
class SpecConfig:
    fft_length: int
    hop_length: int

@dataclass
class OptimizerConfig:
    lr: float
    iter: int
    batch_size: int
    

@dataclass
class _NetworkConfig:
    input_feature: str
    softplus_beta: float

@dataclass
class ModelConfig:
    clean_net: _NetworkConfig
    noise_net: _NetworkConfig
    mixing_type: str

    def _expand_dict_to_class(self, dict_: dict, class_: type):
        return class_(**dict_)
    
    def __post_init__(self):
        self.clean_net = self._expand_dict_to_class(self.clean_net, _NetworkConfig)
        self.noise_net = self._expand_dict_to_class(self.noise_net, _NetworkConfig)

@dataclass
class KurtlossConfig:
    weight: float
    kernel_size: Tuple[int, int] # (Freq., Time)
    stride: Tuple[int, int] # (Freq., Time)

@dataclass
class LossConfig:
    reconstruction_loss_type: str
    clean_segkurtosis_increasing: KurtlossConfig # corresponding to $L_1^{(S)}$
    noise_segkurtosis_decreasing: KurtlossConfig # corresponding to $L_{kurt}^{(N)}$
    clean_refinement_decreasing: KurtlossConfig # corresponding to first term of $L_{2}^{(S)}$
    clean_refinement_increasing: KurtlossConfig # corresponding to second term of $L_{2}^{(S)}$

    def _expand_dict_to_class(self, dict_: dict, class_: type):
        return class_(**dict_)
    
    def __post_init__(self):
        self.clean_segkurtosis_increasing = self._expand_dict_to_class(self.clean_segkurtosis_increasing, KurtlossConfig)
        self.noise_segkurtosis_decreasing = self._expand_dict_to_class(self.noise_segkurtosis_decreasing, KurtlossConfig)
        self.clean_refinement_decreasing = self._expand_dict_to_class(self.clean_refinement_decreasing, KurtlossConfig)
        self.clean_refinement_increasing = self._expand_dict_to_class(self.clean_refinement_increasing, KurtlossConfig)

@dataclass
class Config:
    seed: int
    logging_: LoggingConfig
    preprocess: PreprocessConfig
    spec: SpecConfig
    optimizer: OptimizerConfig
    model: ModelConfig
    loss: LossConfig

    def _expand_dict_to_class(self, dict_: dict, class_: type):
        return class_(**dict_)

    def __post_init__(self):
        self.logging_ = self._expand_dict_to_class(self.logging_, LoggingConfig)
        self.preprocess = self._expand_dict_to_class(self.preprocess, PreprocessConfig)
        self.spec = self._expand_dict_to_class(self.spec, SpecConfig)
        self.optimizer = self._expand_dict_to_class(self.optimizer, OptimizerConfig)
        self.model = self._expand_dict_to_class(self.model, ModelConfig)
        self.loss = self._expand_dict_to_class(self.loss, LossConfig)
        self.preprocess.dump_dir = Path(self.preprocess.dump_dir) \
            if not isinstance(self.preprocess.dump_dir, Path) else self.preprocess.dump_dir
        for i, noise in enumerate(self.preprocess.noise_list):
            self.preprocess.noise_list[i] = Path(noise) \
                if not isinstance(noise, Path) else noise
        for i, clean in enumerate(self.preprocess.clean_list):
            self.preprocess.clean_list[i] = Path(clean) \
                if not isinstance(clean, Path) else clean
        self.logging_.log_dir = Path(self.logging_.log_dir) \
            if not isinstance(self.logging_.log_dir, Path) else self.logging_.log_dir      
        # check choices
        assert self.model.clean_net.input_feature in ["uniform", "uniform-TFcoherent", "meshgrid"]
        assert self.model.noise_net.input_feature in ["uniform", "uniform-TFcoherent", "meshgrid"]
        assert self.loss.reconstruction_loss_type in ["MAE", "MSE"] # "time-MSE", "time-MAE"]
        assert self.model.mixing_type in ["mean", "median", "min"]

    
import torchaudio
import argparse
from pathlib import Path
import random
from skddp_se.utils.config import Config, set_seed
import yaml
import os
import numpy as np
from tqdm import tqdm
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, default="./configs/proposed.yaml")
    args = parser.parse_args()
    config = yaml.load(open(args.yaml_path, "r"), Loader=yaml.FullLoader)
    config = Config(**config)
    set_seed(config.seed)
    (config.preprocess.dump_dir / "noisy").mkdir(parents=True, exist_ok=True)
    (config.preprocess.dump_dir / "clean").mkdir(parents=True, exist_ok=True)
    (config.preprocess.dump_dir / "noise").mkdir(parents=True, exist_ok=True)
    # output log.yaml
    with open(config.preprocess.dump_dir / "log.yaml", "w") as f:
        yaml.dump(config.__dict__, f)

    # load data
    clean_list = []
    for clean_path in config.preprocess.clean_list:
        if os.path.isdir(clean_path):
            clean_list.extend(list(Path(clean_path).glob("*.wav")))
        elif os.path.isfile(clean_path):
            clean_list.append(clean_path)
        else:
            raise ValueError(f"Invalid path: {clean_path}")
    noise_list = []
    for noise_path in config.preprocess.noise_list:
        if os.path.isdir(noise_path):
            noise_list.extend(list(Path(noise_path).glob("*.wav")))
        elif os.path.isfile(noise_path):
            noise_list.append(noise_path)
        else:
            raise ValueError(f"Invalid path: {noise_path}")
    
    config.preprocess.clean_samples = len(clean_list) if config.preprocess.clean_samples is None else config.preprocess.clean_samples
    metadata = []
    bar = tqdm(total=config.preprocess.clean_samples)
    for i in range(config.preprocess.clean_samples):
        clean_path = random.choice(clean_list)
        noise_path = random.choice(noise_list)
        clean_waveform, src = torchaudio.load(clean_path)
        if src != config.preprocess.sample_rate:
            clean_waveform = torchaudio.functional.resample(clean_waveform, src, config.preprocess.sample_rate)
        noise_waveform, srn = torchaudio.load(noise_path)
        if srn != config.preprocess.sample_rate:
            noise_waveform = torchaudio.functional.resample(noise_waveform, srn, config.preprocess.sample_rate)
        clean_waveform = clean_waveform
        noise_waveform = noise_waveform
        if clean_waveform.size(1) < noise_waveform.size(1):
            start_pos = random.randint(0, noise_waveform.size(1) - clean_waveform.size(1))
            noise_waveform = noise_waveform[:, start_pos:start_pos + clean_waveform.size(1)]
        elif clean_waveform.size(1) > noise_waveform.size(1):
            repeat = clean_waveform.size(1) // noise_waveform.size(1)
            remain = clean_waveform.size(1) % noise_waveform.size(1)
            noise_waveform = noise_waveform.repeat(1, repeat)
            noise_waveform = torch.cat([noise_waveform, noise_waveform[:, :remain]], dim=1)
        else:
            clean_waveform = clean_waveform[:, :noise_waveform.size(1)]
        SNR = random.choice(config.preprocess.SNR_list)
        SNR = torch.tensor([SNR])
        noisy_waveform = torchaudio.functional.add_noise(clean_waveform, noise_waveform, SNR)
        torchaudio.save(config.preprocess.dump_dir / "clean" / f"{i}.wav", clean_waveform, config.preprocess.sample_rate)
        torchaudio.save(config.preprocess.dump_dir / "noise" / f"{i}.wav", noise_waveform, config.preprocess.sample_rate)
        np.save(config.preprocess.dump_dir / "noisy" / f"{i}.npy", noisy_waveform.numpy()) # avoid clipping/auto-normalization
        metadata.append("|".join(
            [str(i), str(clean_path), str(noise_path), str(SNR.item())]
        ))
        bar.update(1)
    bar.close()
    with open(config.preprocess.dump_dir / "metadata.txt", "w") as f:
        f.write("\n".join(metadata))
    print("Data preparation is done.")

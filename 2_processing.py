import argparse
from pathlib import Path
import random
from skddp_se.utils.config import Config, set_seed
import yaml
import torch
from skddp_se.model.simple_unet import SimpleTwinUnet
from skddp_se.model.loss import skddpLoss
import numpy as np
import json
from logging import getLogger, config
import torchaudio
import einops

with open('./skddp_se/utils/logging.json', 'r') as f:
    log_conf = json.load(f)
config.dictConfig(log_conf)

if __name__ == "__main__":
    # --- setting ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, default="./configs/proposed.yaml")
    parser.add_argument("--device_id", type=int, default=0) # if -1, use cpu
    args = parser.parse_args()
    config = yaml.load(open(args.yaml_path, "r"), Loader=yaml.FullLoader)
    config = Config(**config)
    set_seed(config.seed)
    [(config.logging_.log_dir/aaa).mkdir(parents=True, exist_ok=True) for aaa in ["clean", "noise", "noisy"]]
    logger = getLogger(__name__)
    # output log.yaml
    with open(config.logging_.log_dir / "log.yaml", "w") as f:
        yaml.dump(config.__dict__, f)
    # --- processing ---
    DEVICE = torch.device("cuda", args.device_id) if args.device_id >= 0 else torch.device("cpu")
    processed_samples = list((config.preprocess.dump_dir / "noisy").glob("*.npy"))
    STFT = torchaudio.transforms.Spectrogram(
        n_fft=config.spec.fft_length,
        win_length=config.spec.fft_length,
        hop_length=config.spec.hop_length,
        power=None
    ).to(DEVICE)
    ISTFT = torchaudio.transforms.InverseSpectrogram(
        n_fft=config.spec.fft_length,
        win_length=config.spec.fft_length,
        hop_length=config.spec.hop_length
    ).to(DEVICE)
    for processed_sample in processed_samples:
        logger.info(f"Processing {processed_sample.stem}")
        noisy_sample = np.load(processed_sample)
        noisy_sample = torch.from_numpy(noisy_sample).to(DEVICE)
        noisy_spec = STFT(noisy_sample)
        noisy_amp, noisy_angle = torch.abs(noisy_spec), torch.angle(noisy_spec)
        noisy_amp = einops.rearrange(noisy_amp, "1 freq time -> 1 1 freq time")
        model = SimpleTwinUnet(config.model).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)
        loss = skddpLoss(config.loss, noisy_amp).to(DEVICE)
        noisy_amp = einops.repeat(noisy_amp, "1 1 freq time -> batch 1 freq time", batch=config.optimizer.batch_size)
        z1, z2 = model.get_input_features(noisy_amp)
        log_step = config.optimizer.iter // config.logging_.log_freq
        with torch.no_grad():
            clean_est, clean_est_merged, noise_est = model(z1, z2)
            loss_value, loss_logits = loss(clean_est, clean_est_merged, noise_est, noisy_amp)
            logger_output = "iter: 0, "
            for key, value in loss_logits.items():
                logger_output += f"{key}: {value:.3e}, "
            logger.info(logger_output)
        for i in range(config.optimizer.iter):
            optimizer.zero_grad()
            clean_est, clean_est_mixed, noise_est = model(z1, z2)
            loss_value, loss_logits = loss(clean_est, clean_est_merged, noise_est, noisy_amp)
            loss_value.backward()
            optimizer.step()
            # logging
            if (i+1) % log_step == 0:
                logging_output = f"iter: {i+1}, "
                for key, value in loss_logits.items():
                    logging_output += f"{key}: {value:.3e}, "
                logger.info(logging_output)
            # save
            if (i+1) in config.logging_.save_intermediate_steps:
                with torch.no_grad():
                    # clean
                    clean_est_mixed = einops.rearrange(clean_est_mixed, "1 1 freq time -> 1 freq time")
                    clean_est_mixed = clean_est_mixed * torch.exp(1j*noisy_angle)
                    clean_est_mixed = ISTFT(clean_est_mixed)
                    clean_est_mixed = clean_est_mixed.detach().cpu()
                    torchaudio.save(
                        config.logging_.log_dir / "clean" / f"{processed_sample.stem}_{i+1}.wav",
                        clean_est_mixed,
                        config.preprocess.sample_rate
                    )
                    # noise
                    noise_est = einops.rearrange(noise_est, "1 1 freq time -> 1 freq time")
                    noise_est = noise_est * torch.exp(1j*noisy_angle)
                    noise_est = ISTFT(noise_est)
                    noise_est = noise_est.detach().cpu()
                    torchaudio.save(
                        config.logging_.log_dir / "noise" / f"{processed_sample.stem}_{i+1}.wav",
                        noise_est,
                        config.preprocess.sample_rate
                    )
                    # noisy
                    noisy_est = clean_est_mixed + noise_est
                    torchaudio.save(
                        config.logging_.log_dir / "noisy" / f"{processed_sample.stem}_{i+1}.wav",
                        noisy_est,
                        config.preprocess.sample_rate
                    )
        # save final
        with torch.no_grad():
            # clean
            clean_est_mixed = einops.rearrange(clean_est_mixed, "1 1 freq time -> 1 freq time")
            clean_est_mixed = clean_est_mixed * torch.exp(1j*noisy_angle)
            clean_est_mixed = ISTFT(clean_est_mixed)
            clean_est_mixed = clean_est_mixed.detach().cpu()
            torchaudio.save(
                config.logging_.log_dir / "clean" / f"{processed_sample.stem}_{config.optimizer.iter}.wav",
                clean_est_mixed,
                config.preprocess.sample_rate
            )
            # noise
            noise_est = einops.rearrange(noise_est, "1 1 freq time -> 1 freq time")
            noise_est = noise_est * torch.exp(1j*noisy_angle)
            noise_est = ISTFT(noise_est)
            noise_est = noise_est.detach().cpu()
            torchaudio.save(
                config.logging_.log_dir / "noise" / f"{processed_sample.stem}_{config.optimizer.iter}.wav",
                noise_est,
                config.preprocess.sample_rate
            )
            # noisy
            noisy_est = clean_est_mixed + noise_est
            torchaudio.save(
                config.logging_.log_dir / "noisy" / f"{processed_sample.stem}_{config.optimizer.iter}.wav",
                noisy_est,
                config.preprocess.sample_rate
            )
    logger.info("Finished.")
from pystoi import stoi
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality as pesq
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as sisdr
import torchaudio
from logging import getLogger, config, Logger
import argparse
import yaml
from skddp_se.utils.config import Config, set_seed
from pathlib import Path
import json
import polars
import torch
import numpy as np

with open('./skddp_se/utils/logging.json', 'r') as f:
    log_conf = json.load(f)
config.dictConfig(log_conf)

def metrics(eval_target_path, eval_pred_path, sr=16000, logger: Logger = None):
    if eval_target_path.suffix == ".npy":
        target_waveform = torch.from_numpy(np.load(eval_target_path))
    else:
        target_waveform, src = torchaudio.load(eval_target_path)
        if src != sr:
            if logger is not None:
                logger.warning(f"Resampling target waveform from {src} to {sr}")
            target_waveform = torchaudio.functional.resample(target_waveform, src, sr)
    if eval_pred_path.suffix == ".npy":
        pred_waveform = torch.from_numpy(np.load(eval_pred_path))
    else:
        pred_waveform, srn = torchaudio.load(eval_pred_path)
        if srn != sr:
            if logger is not None:
                logger.warning(f"Resampling predicted waveform from {srn} to {sr}")
            pred_waveform = torchaudio.functional.resample(pred_waveform, srn, sr)
    pesq_score = pesq(
        preds=pred_waveform,
        target=target_waveform,
        fs=sr,
        mode="wb"
    )
    sisdr_score = sisdr(
        preds=pred_waveform,
        target=target_waveform
    )
    estoi_score = stoi(
        target_waveform.squeeze().numpy(),
        pred_waveform.squeeze().numpy(),
        fs_sig=sr,
        extended=True
    )
    return pesq_score, sisdr_score, estoi_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.load(open(args.log_dir/"log.yaml", "r"), Loader=yaml.FullLoader)
    config = Config(**config)
    set_seed(config.seed)
    logger = getLogger(__name__)
    logger.info(f"Start evaluation")
    results = {
        "eval_target_path": [],
        "pesq_post": [],
        "pesq_pre": [],
        "sisdr_post": [],
        "sisdr_pre": [],
        "estoi_post": [],
        "estoi_pre": []
    }
    for eval_target_path in (config.logging_.log_dir / "clean").glob(f"*_{config.optimizer.iter}.wav"):
        stem_number, stem_iter = eval_target_path.stem.split("_")
        eval_pred_path = config.preprocess.dump_dir / "clean" / f"{stem_number}.wav"
        noisy_path = config.preprocess.dump_dir / "noisy" / f"{stem_number}.npy"
        pesq_score_post, sisdr_score_post, estoi_score_post = \
            metrics(eval_target_path, eval_pred_path, logger=logger)
        pesq_score_pre, sisdr_score_pre, estoi_score_pre = \
            metrics(eval_target_path, noisy_path, logger=logger)
        results["eval_target_path"].append(str(eval_target_path))
        results["pesq_post"].append(pesq_score_post)
        results["pesq_pre"].append(pesq_score_pre.item())
        results["sisdr_post"].append(sisdr_score_post)
        results["sisdr_pre"].append(sisdr_score_pre.item())
        results["estoi_post"].append(estoi_score_post)
        results["estoi_pre"].append(estoi_score_pre.item())
    results = polars.DataFrame(results)
    results.write_csv(args.log_dir / "evaluation.csv")
    logger.info(f"Finish evaluation")

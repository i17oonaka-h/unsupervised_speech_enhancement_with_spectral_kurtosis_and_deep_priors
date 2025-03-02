# Details of configuration

## Seed (int): Seed for random number generator

## logging_: Logging configuration
### log_dir (str): Directory to save logs and results
### log_freq (int): Logging frequency. `log_step = optimizer.iter // log_freq`
### save_intermediate_steps (List[int]): List of iteration to save intermediate results

## preprocess: Preprocessing configuration
### dump_dir (str): Directory to save preprocessed data.
    - dump_dir / "noisy": Directory to save noisy waveform of .npy format (avoid clipping/auto-normalization)
    - dump_dir / "clean": Directory to save clean waveform
    - dump_dir / "noise": Directory to save noise waveform
    - metadata.txt: Metadata of preprocessed data (file_stem|clean_path|noise_path|SNR)
    - log.yaml: Copy of this configuration
### clean_list (List[str]): List of clean data path. If directory, all .wav files in the directory will be used.
### noise_list (List[str]): List of noise data path. If directory, all .wav files in the directory will be used.
### clean_samples (int): Number of samples to preprocess
### clip_seconds (int or None): Clip seconds of waveform. If None, use the original clean waveform length.

## spec: Spectrogram configuration
### fft_length (int): FFT length of STFT
### hop_length (int): Hop length of STFT

## optimizer: Optimizer configuration
### lr (float): Learning rate
### iter (int): Number of iteration
### batch_size (int): Batch size

## model: Model configuration
### mixing_type (str): Mixing type of predicted clean spectrograms. "mean", "median", or "min"
### clean_net: Configuration of clean spectrogram estimation network
    - input_feature (str): Input feature of clean spectrogram estimation network. "uniform", "uniform-TFcoherent", or "meshgrid"
    - softplus_beta (float): Beta of softplus layer
### noise_net: Configuration of noise spectrogram estimation network
    - input_feature (str): Input feature of noise spectrogram estimation network. "uniform", "uniform-TFcoherent", or "meshgrid"
    - softplus_beta (float): Beta of softplus layer

## loss: Loss configuration
### reconstruction_loss_type (str): Reconstruction loss type. "MSE", "MAE"
### clean_segkurtosisi_increasing: Configuration of clean segmental kurtosis increasing loss (corresponding to `L_1^(S)`)
    - weight (float): Weight of clean segmental kurtosis increasing loss
    - kernel_size (List[int]): Kernel size of segmental kurtosis increasing loss. `[freq, time]`
    - stride (List[int]): Stride of segmental kurtosis increasing loss. `[freq, time]`
### noise_segkurtosis_decresing: Configuration of noise segmental kurtosis decreasing loss (corresponding to `L_{kurt}^(N)`)
    - same as clean_segkurtosisi_increasing
### clean_refinement_decreasing: Configuration of clean refinement decreasing loss (corresponding to `1st-term of L_{2}^(S)`)
    - same as clean_segkurtosisi_increasing
### clean_refinement_increasing: Configuration of clean refinement increasing loss (corresponding to `2nd-term of L_{2}^(S)`)
# unsupervised_speech_enhancement_with_spectral_kurtosis_and_deep_priors
The official implementation of "Unsupervised speech enhancement with spectral kurtosis and double deep priors."

## 1. Setup
Install the package `skddp_se` and necessary libraries (refer to `install_requires in setup.py`) with the following command.
```bash
pip install -e .
```

If necessary, call the corpus download script to run the code. This will download the LJSpeech [1] and DEMAND noise databases [2] to `./corpus`.
```bash
bash 0_corpus_download.sh 
```

# Examples
- example_segkurtosis.ipynb
    - Sample code to randomly select an audio/noise file downloaded by setup and output its kurtosis.
- example_speech_enhancement.ipynb
    - Sample code of our speech enhancement method.

## Dataset processing
### 1. Preprocessing
Executing the following code will generate noisy speech as a preprocessing step.
The various configurations of this code are managed by yaml and its wrapper, dataclass (`skddp_se/utils/config.py`), which refers to `configs/proposed.yaml` by default.
```bash
python 1_data_prepare.py
# or
# python 1_data_prepare.py --yaml_path [your configuration]
```

### 2. Speech enhancement
Apply speech enhancement to the noisy speech generated in step 2.
```bash
python 2_processing.py
# option
# --yaml_path [your configuration]
# --device_id [device_id] # if -1, use cpu
```

### 3. Calculate metrics
```bash
python 3_metrics.py --log_dir [log_dir of step 3]
```
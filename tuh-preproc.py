import numpy as np
from pathlib import Path
import mne
from braindecode.datasets import TUHAbnormal
from braindecode.preprocessing import (preprocess, Preprocessor, scale)

mne.set_log_level('ERROR') #Remove extraneous mne messages
n_jobs = -1 #Set proprocessor to use all cores

# Set data load path
cwd = Path.cwd()
load_path = cwd / 'TUH_Raw/'

# Load data from TUH Abnormal EEG corpus
tuh = TUHAbnormal(
    path = load_path,
    recording_ids = None,
    target_name = 'pathological',
    preload = False,
    add_physician_reports = False,
)

# desc = tuh.description

# Create preprocessing pipeline for data
chs = tuh.datasets[0].raw.info.ch_names[0:21]
preprocessors = [
    Preprocessor('pick',picks=chs),
    Preprocessor('set_eeg_reference', ref_channels=['EEG A1-REF','EEG A2-REF'], ch_type='eeg'),
    Preprocessor(scale, factor=1e6, apply_on_array=True),
    Preprocessor(np.clip, a_min=-800, a_max=800, apply_on_array=True),
    Preprocessor('resample', sfreq=100)
]

# Save preprocessed data
save_path = cwd / 'TUH_Preprocessed/'
tuh_preproc = preprocess(
    concat_ds = tuh,
    preprocessors = preprocessors,
    n_jobs = n_jobs,
    save_dir = save_path,
    overwrite = True
)

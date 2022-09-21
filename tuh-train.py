import mne
from pathlib import Path
from braindecode.preprocessing import (preprocess, Preprocessor, create_fixed_length_windows)
from braindecode.datautil import load_concat_dataset

mne.set_log_level('ERROR') #Remove extraneous mne messages
n_jobs = -1 #Set proprocessor to use all cores

# Set load path
cwd = Path.cwd()
load_path = cwd / 'TUH_Preprocessed/'

# Load preprocessed data
tuh_preproc = load_concat_dataset(
    path = load_path,
    preload = True,
    target_name = 'pathological',
)

# Define crop function and set min and max times
def custom_crop(raw, tmin=0.0, tmax=None, include_tmax=True):
    # crop recordings to tmin – tmax. can be incomplete if recording
    # has lower duration than tmax
    # by default mne fails if tmax is bigger than duration
    tmax = min((raw.n_times - 1) / raw.info['sfreq'], tmax)
    raw.crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)
tmin = 1*60
tmax = 2*60

# Crop preprocessed data
tuh_cropped = preprocess(
    concat_ds = tuh_preproc,
    preprocessors = [Preprocessor(custom_crop, tmin=tmin, tmax=tmax, apply_on_array=False)],
    n_jobs = n_jobs
)

# Generate compute windows
window_size_samples = 1000
window_stride_samples = 1000
tuh_windows = create_fixed_length_windows(
    tuh_cropped,
    window_size_samples = window_size_samples,
    window_stride_samples = window_stride_samples,
    drop_last_window = False,
    n_jobs = n_jobs,
)
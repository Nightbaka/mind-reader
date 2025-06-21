import mne
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

ROOT_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ROOT_RAW_DIR = os.path.join(ROOT_DATA_DIR, 'raw')
TRAIN_DATA_DIR = os.path.join(ROOT_DATA_DIR, 'preprocessed')

def load_raw_data(file_path=ROOT_RAW_DIR, task='sternberg') -> dict[str, mne.io.Raw]:
    """Load raw EEG data from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    print(f"Loading raw data from {file_path}")
    # get all the subdirectories with sub-n prefix
    subdirs = [d for d in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, d)) and d.startswith('sub-')]
    print(f"Found {len(subdirs)} subjects in {file_path}")
    subjects = {}
    for subdir in subdirs:
        print(f"Processing subject: {subdir}. file_path: {file_path}")
        sub_path = os.path.join(file_path, subdir, 'eeg')
        header_files = [f for f in os.listdir(sub_path) if f.endswith(f'{task}_eeg.vhdr')]
        for file in header_files:
            subject_file_path = os.path.join(sub_path, file)
            print(f"Loading raw data from {subject_file_path}")
            raw = mne.io.read_raw_brainvision(subject_file_path, preload=False)
            subjects[subdir] = raw
    return subjects


def preprocess_data(
    raw: mne.io.Raw,
    target_approx_sfreq: float = 250.0,
    bandpass: tuple[float, float] = (30.0, 50.0),
    notch_filter: list[float] = [50.0, 100.0],
    channels: list[str] = [
        "Fp1", "Fp2", "F3", "F4",
        "O1", "O2",
        "T7", "T8",
        "P3", "P4",
        "C1", "C2",
        "F10", "F9",
        "P9", "P10"
    ]
) -> mne.io.Raw:
    """
    Apply preprocessing to a single Raw object:
      1) resample to `sfreq`
      2) band-pass filter between bandpass[0]–bandpass[1] Hz
      3) notch filter at frequencies in notch_filter
      4) re-reference to common average
    """
    raw.pick_channels(channels)
    raw = raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], fir_design="firwin")
    raw.notch_filter(freqs=notch_filter, fir_design="firwin")
    original_sfreq = raw.info["sfreq"]
    if original_sfreq != target_approx_sfreq:
        raw.resample(target_approx_sfreq, npad="auto", verbose=False)
    else:
        print(f"Raw data already at target sampling frequency: {target_approx_sfreq} Hz")
    raw.set_eeg_reference("average", projection=False)

    return raw

def load_to_preprocessed(
    subjects: dict[str, mne.io.Raw],
    target_approx_sfreq: float = 250.0,
    bandpass: tuple[float, float] = (30.0, 50.0),
    notch_filter: list[float] = [50.0, 100.0],
    save_path: str = TRAIN_DATA_DIR,
) -> mne.io.Raw:
    """
    Load and preprocess the raw data.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for subj, raw in subjects.items():
        print(f"Preprocessing data for {subj}")
        raw.load_data() 
        raw = preprocess_data(raw, target_approx_sfreq, bandpass, notch_filter)
        preprocessed_file = os.path.join(save_path, f"{subj}_preprocessed_raw.fif")
        raw.save(preprocessed_file, overwrite=True)


def main():
    subjects = load_raw_data()
    print(f"Raw data loaded of {len(subjects)} subjects. Starting preprocessing...")
    load_to_preprocessed(subjects)
    print("Preprocessing complete.")

if __name__ == '__main__':
    main()

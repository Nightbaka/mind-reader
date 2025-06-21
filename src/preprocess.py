import mne
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

ROOT_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ROOT_RAW_DIR = os.path.join(ROOT_DATA_DIR, 'raw')
TRAIN_DATA_DIR = os.path.join(ROOT_DATA_DIR, 'preprocessed')

def fix_dataset():
    '''
    Dataset after downloading has some errors in th efile names in header files. Filepaths must be corrected in subject 29 and 57 filepaths
    '''
    subject_29_path = os.path.join(
        ROOT_RAW_DIR,
        "sub-29",
        "eeg",
        "sub-29_task-sternberg_eeg.vhdr",
    )
    with open(subject_29_path, 'r') as f:
        lines = f.readlines()
    lines_cp = lines.copy()
    lines_cp[5] = "DataFile=sub-29_task-sternberg_eeg.eeg\n"
    lines_cp[6] = "MarkerFile=sub-29_task-sternberg_eeg.vmrk\n"
    if lines[5] != lines_cp[5] or lines[6] != lines_cp[6]:
        print(f"Fixing subject 29 file")
        with open(subject_29_path, 'w') as f:
            f.writelines(lines_cp)

    # Fix subject 57
    subject_57_path = os.path.join(
        ROOT_RAW_DIR,
        "sub-57",
        "eeg",
        "sub-57_task-sternberg_eeg.vhdr",
    )
    with open(subject_57_path, 'r') as f:
        lines = f.readlines()

    lines_cp = lines.copy()
    lines_cp[5] = "DataFile=sub-57_task-sternberg_eeg.eeg\n"
    lines_cp[6] = "MarkerFile=sub-57_task-sternberg_eeg.vmrk\n"
    if lines[5] != lines_cp[5] or lines[6] != lines_cp[6]:
        print(f"Fixing subject 57 file")
        with open(subject_57_path, 'w') as f:
            f.writelines(lines_cp)

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
    target_approx_sfreq: float = 256.0,
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


def epoch_data(
    raw: mne.io.Raw, epoch_length_sec: float = 1.0, preload: bool = True
) -> mne.Epochs:
    """
    Divide continuous raw data into fixed-length epochs.
    epoch_length_sec: length of each epoch in seconds (default 1.0s)
    """
    # create synthetic events every `epoch_length_sec`
    events = mne.make_fixed_length_events(raw, duration=epoch_length_sec)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=epoch_length_sec,
        baseline=None,
        preload=preload,
        verbose=False,
    )
    return epochs


def load_to_preprocessed(
    subjects: dict[str, mne.io.Raw],
    target_approx_sfreq: float = 250.0,
    bandpass: tuple[float, float] = (30.0, 50.0),
    notch_filter: list[float] = [50.0, 100.0],
    epoch_length_sec: float = 1.0,
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
        epochs = epoch_data(raw, epoch_length_sec=epoch_length_sec)
        preprocessed_file = os.path.join(save_path, f"{subj}_preprocessed_raw.fif")
        epochs.save(preprocessed_file, overwrite=True)


def main():
    subjects = load_raw_data()
    print(f"Raw data loaded of {len(subjects)} subjects. Starting preprocessing...")
    load_to_preprocessed(subjects)
    print("Preprocessing complete.")

if __name__ == '__main__':
    main()

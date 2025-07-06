"""
EEG Preprocessing Script for BCI Competition III - Dataset V
This script prepares EEG signals for generative model training and classification.
It includes filtering, artifact annotation, and ICA-based artifact correction.

"""

import zipfile
import scipy.io
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA

# Extract data if not already extracted
zip_files = [
    "dataset/BCI_3/subject1_mat.zip",
    "dataset/BCI_3/subject2_mat.zip",
    "dataset/BCI_3/subject3_mat.zip"
]

for zip_path in zip_files:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("dataset/BCI_3/")

# Load MATLAB EEG data for Subject 1
train_data = scipy.io.loadmat('dataset/BCI_3/train_subject1_raw01.mat')
test_data = scipy.io.loadmat('dataset/BCI_3/test_subject1_raw04.mat')

# Inspect the keys of the loaded data (for reference/debug)
print(train_data.keys())
print(test_data.keys())

# Extract EEG signals
eeg_train = train_data['X']
eeg_test = test_data['X']

# Extract labels if present
labels_train = train_data.get('Y', None)

print("EEG train shape:", eeg_train.shape)
print("EEG test shape:", eeg_test.shape)
if labels_train is not None:
    print("Train labels shape:", labels_train.shape)

# Channel names according to the 10-20 system
channel_names = [
    'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5',
    'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8',
    'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2',
    'Fz', 'Cz'
]

# Sampling frequency
sfreq = 512

# Create MNE Info and RawArray
info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
raw_train = mne.io.RawArray(eeg_train.T, info)
raw_test = mne.io.RawArray(eeg_test.T, info)

# Plot raw EEG
raw_train.plot(scalings='auto', title='Raw EEG - Subject 1 (Train)')

# Bandpass filter: 8–30 Hz (Mu and Beta rhythms)
raw_train_filt = raw_train.copy().filter(l_freq=8., h_freq=30., fir_design='firwin')
raw_test_filt = raw_test.copy().filter(l_freq=8., h_freq=30., fir_design='firwin')

# Plot filtered EEG
raw_train_filt.plot(scalings='auto', title='Filtered EEG (8–30 Hz) - Subject 1')

# Annotate muscle artifacts using z-score method
annotations_train, _ = mne.preprocessing.annotate_muscle_zscore(
    raw_train_filt, ch_type='eeg', threshold=4.0, min_length_good=0.1, filter_freq=[20, 140]
)

annotations_test, _ = mne.preprocessing.annotate_muscle_zscore(
    raw_test_filt, ch_type='eeg', threshold=4.0, min_length_good=0.1, filter_freq=[20, 140]
)

# Apply annotations
raw_train_filt.set_annotations(annotations_train)
raw_test_filt.set_annotations(annotations_test)

# Plot with artifacts
raw_train_filt.plot(title='Annotated Artifacts - Train')

# Independent Component Analysis (ICA)
ica_train = ICA(n_components=20, random_state=97, max_iter=800)
ica_test = ICA(n_components=20, random_state=97, max_iter=800)

ica_train.fit(raw_train_filt)
ica_test.fit(raw_test_filt)

# Automatically find EOG-related components
eog_idx_train, _ = ica_train.find_bads_eog(raw_train_filt, ch_name=['Fp1', 'Fp2'])
eog_idx_test, _ = ica_test.find_bads_eog(raw_test_filt, ch_name=['Fp1', 'Fp2'])

# Mark EOG components for exclusion
ica_train.exclude = eog_idx_train
ica_test.exclude = eog_idx_test

# Apply ICA to remove eye-blink artifacts
raw_train_clean = raw_train_filt.copy()
raw_test_clean = raw_test_filt.copy()

ica_train.apply(raw_train_clean)
ica_test.apply(raw_test_clean)

# Plot corrected EEG
raw_train_clean.plot(scalings='auto', title='ICA Corrected EEG - Train')

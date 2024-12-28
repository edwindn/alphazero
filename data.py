import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import sys
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchaudio.models import wav2vec2_base
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_FILES = 20
assert len(sys.argv) == 2, 'Please specify the current run'
run = int(sys.argv[1])

if run == 100:
    files = [f'data/dataloader_{i}_wav2vec.pth' for i in range(39)] # up to 38
    train_loaders = []
    for file in files:
        data = torch.load(file)
        train_loaders.append(data.dataset)
    
    train_loader = ConcatDataset(train_loaders)
    train_loader = DataLoader(train_loader, shuffle=True, batch_size=16)
    test_loader = torch.load('data/dataloader_39_wav2vec.pth')
    torch.save(train_loader, 'data/train_loader_wav2vec.pth')
    torch.save(test_loader, 'data/test_loader_wav2vec.pth')
    quit()

wav2vec = wav2vec2_base().to(device)
processor = wav2vec.feature_extractor
encoder = wav2vec.encoder

def load_wav_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

yes_data = []
no_data = []
sr = 44100

lens = []

print('Extracting audio files...')
for i in range(400):
    no, _ = load_wav_file(f'yesno_voice_recognition/train/no{i}.wav')
    no = librosa.resample(no, orig_sr=44100, target_sr=16000) # optional for direct model
    lens.append(no.shape[0])
    no_data.append((no, 0))
    yes, _ = load_wav_file(f'yesno_voice_recognition/train/yes{i}.wav')
    yes = librosa.resample(yes, orig_sr=44100, target_sr=16000) # optional for direct model
    lens.append(yes.shape[0])
    yes_data.append((yes, 1))

data = yes_data + no_data

fixed_len  = 60000
for i, (d, l) in enumerate(data):
    if d.shape[0] < fixed_len:
        d = np.pad(d, (0, fixed_len - d.shape[0]), 'constant')
    else:
        d = d[:fixed_len]
    data[i] = (d, l)

data = [(torch.tensor(d, dtype=torch.float32).unsqueeze(0), torch.tensor(l)) for d, l in data]

print('Encoding files...')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
wav2vec_train = []
wav2vec_test = []
def process_audio(data):
    processed = []
    for d in tqdm(data):
        torch.cuda.empty_cache()
        audio, label = d
        audio = audio.to(device)
        #with autocast():
        audio = processor(audio, length=fixed_len)[0]
        audio = encoder(audio)
        processed.append((audio, label))
    return processed

wav2vec_train = process_audio(data[run*MAX_FILES:min(len(data), (run+1)*MAX_FILES)])

dataloader = DataLoader(wav2vec_train, shuffle=True, batch_size=16)

torch.save(dataloader, f'data/dataloader_{run}_wav2vec.pth')
print('Files saved successfully.')

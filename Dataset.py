import random
import os
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

def _trim(audio, sr):
    trim_start = random.randint(0, len(audio)-sr)
    trim_end = trim_start+sr+1 # add a second and include last sample
    audio_trimmed = audio.copy()
    audio_trimmed[trim_start:trim_end] = [0] * (trim_end - trim_start)
    return audio_trimmed

def _load_wav(filename, sample_rate=22050, trim_flag=False):
    """
    Read all wav files in data dir and transform to a func of frequency in time
    Trim all audios to 5sec
    Return data (list)
    """

    audio, sr = librosa.load(filename, sr=sample_rate)
    if len(audio)/sr < 5:
        print(filename, "less than 5 secs")
    else:
        audio = audio[:(5*sr)+1]
        stft_audio = librosa.feature.melspectrogram(y = audio, sr = sample_rate)
        audio_tensor = torch.from_numpy(stft_audio).unsqueeze(0)
        # amp = audio_tensor**2
        # data_2c = torch.concat((audio_tensor.unsqueeze(0), amp.unsqueeze(0)),dim=0)
        
        if trim_flag:
            trimmed = _trim(audio, sr)
            stft_audio_trimmed = librosa.feature.melspectrogram(y = trimmed, sr = sample_rate)
            audio_trimmed_tensor = torch.from_numpy(stft_audio_trimmed).unsqueeze(0)
            # amp = audio_trimmed_tensor**2
            # data_2c_trimmed = torch.concat((audio_trimmed_tensor.unsqueeze(0), amp.unsqueeze(0)),dim=0)
            return audio_tensor, audio_trimmed_tensor   
        else:
            return audio_tensor
    
class AudioDataset(Dataset):
    """Dataset returning images in a folder."""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        # check files
        supported_formats=['wav']        
        self.files=[el for el in os.listdir(self.root_dir) if el.split('.')[-1] in supported_formats]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):          

        img_name = os.path.join(self.root_dir,
                                self.files[idx])
        
        return _load_wav(img_name)

class AudioDatasetConditional(Dataset):
    """Dataset returning images in a folder."""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        # check files
        supported_formats=['wav']        
        self.files=[el for el in os.listdir(self.root_dir) if el.split('.')[-1] in supported_formats]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):          

        img_name = os.path.join(self.root_dir,
                                self.files[idx])
        
        audio, audio_trimmed = _load_wav(img_name)

        return {"audio": audio, "trimmed": audio_trimmed}
    

def plot_spectrogram(stft_audio_sample, sr = 22050, hop_length = 512, y_axis = "linear", ylog=False, save = False, filename = None):
    # convert to numpy float64
    stft_audio_sample = stft_audio_sample.numpy().astype('float64')
    # get amplitude
    y = np.abs(stft_audio_sample) ** 2

    if ylog:
        y = librosa.power_to_db(y)
        
    plt.figure(figsize = (25,10))
    librosa.display.specshow(y, sr = sr, hop_length = hop_length, x_axis = "time", y_axis = y_axis)
    plt.colorbar(format="%+2.f")
    if save:
        assert(filename != None)
        plt.savefig("./spectogram_imgs/%s" %filename)

    
    
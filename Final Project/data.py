import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset

import vocabulary
from TextTransform import TextTransform
from constants import SR, FILE_2CHECK, N_MFCC, N_MELS
from distances import extract_features


class Data:
    def __init__(self):
        print('Loading data...')
        self.data_dir = 'an4'
        self.x_train_paths, self.x_train, self.y_train = self.load_data('train')
        self.x_val_paths, self.x_val, self.y_val = self.load_data('val')
        self.x_test_paths, self.x_test, self.y_test = self.load_data('test')

        # Print the first example in the training set
        # print(f"First train file path: {self.x_train_paths[FILE_2CHECK]}")
        # print(f"Transcription: {self.y_train[FILE_2CHECK]}")
        # print(f"Preview first train file: {self.x_train[FILE_2CHECK]}")
        # self.plot_waveform(self.x_train[0], sample_rate=SR)
        # self.plot_mfcc(extract_features(self.x_train[FILE_2CHECK]))
        # self.plot_mel_spec(self.extract_mel_spec(self.x_train[FILE_2CHECK]))
        print('Data loaded successfully')

    def load_data(self, split):
        """
        Load the data from the provided 'an4' folder, and split it into train, dev, and test sets.
        """
        audio_dir = os.path.join(self.data_dir, split, 'an4', 'wav')
        transcript_dir = os.path.join(self.data_dir, split, 'an4', 'txt')

        audio_files = sorted(os.listdir(audio_dir))
        transcript_files = sorted(os.listdir(transcript_dir))

        audio_paths = [os.path.join(audio_dir, file) for file in audio_files]
        transcript_paths = [os.path.join(transcript_dir, file) for file in
                            transcript_files]

        audios, transcripts = [], []

        for audio_path, transcript_path in zip(audio_paths, transcript_paths):
            with open(transcript_path, 'r') as f:
                transcript = f.read().strip()
                transcripts.append(transcript)

        loaded_audios = [torchaudio.load(audio)[0] for audio in audio_paths]
        return audio_paths, loaded_audios, transcripts

    def get_data(self, split):
        if split == 'train':
            return self.x_train, self.y_train
        elif split == 'val':
            return self.x_val, self.y_val
        elif split == 'test':
            return self.x_test, self.y_test
        else:
            raise ValueError(f"Invalid data split '{split}'")

    @staticmethod
    def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
            if xlim:
                axes[c].set_xlim(xlim)
            if ylim:
                axes[c].set_ylim(ylim)
        figure.suptitle(title)
        plt.show(block=False)

    @staticmethod
    def extract_mel_spec(waveform):
        mel_specgram = torchaudio.transforms.MelSpectrogram(SR, n_mels=N_MELS)(waveform)
        return mel_specgram

    @staticmethod
    def plot_mfcc(mfcc, title="MFCC", xlim=None, ylim=None):
        fig, ax = plt.subplots()
        im = ax.imshow(mfcc, origin='lower', aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.set(title=title, xlabel='Time', ylabel='MFCC')
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        plt.show(block=False)

    @staticmethod
    def plot_mel_spec(mel_spec, title="Mel Spectrogram", xlim=None, ylim=None):
        mel_spec = mel_spec.squeeze(0).numpy()
        fig, ax = plt.subplots()
        im = ax.imshow(mel_spec, origin='lower', aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.set(title=title, xlabel='Time', ylabel='Frequency (Hz)')
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        plt.show(block=False)


class AN4Dataset(Dataset):
    def __init__(self, split, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        audio_dir = os.path.join('an4', split, 'an4', 'wav')
        transcript_dir = os.path.join('an4', split, 'an4', 'txt')

        audio_files = sorted(os.listdir(audio_dir))
        transcript_files = sorted(os.listdir(transcript_dir))

        audio_paths = [os.path.join(audio_dir, file) for file in audio_files]
        transcript_paths = [os.path.join(transcript_dir, file) for file in
                            transcript_files]

        self.audios, self.transcripts = [], []

        for audio_path, transcript_path in zip(audio_paths, transcript_paths):
            with open(transcript_path, 'r') as f:
                transcript = f.read().strip()
                self.transcripts.append(transcript)

        self.loaded_audios = [torchaudio.load(audio)[0] for audio in audio_paths]

    def __len__(self):
        return len(self.loaded_audios)

    def __getitem__(self, idx):
        return self.loaded_audios[idx], self.transcripts[idx]


def data_processing(data, vocabulary, data_type="train"):
    inputs, inputs_lengths, labels, labels_length = [], [], [], []

    if data_type == "train":
        transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_mels=N_MELS),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35))
    else:
        transform = torchaudio.transforms.MelSpectrogram()

    text_transform = TextTransform()
    for (wav, transcript) in data:
        spec = transform(wav).squeeze(0).transpose(0, 1)
        inputs.append(spec)
        inputs_lengths.append(spec.shape[0] // 2)
        label = torch.Tensor(text_transform.text_to_int(transcript))
        labels.append(label)
        labels_length.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(inputs, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return spectrograms, labels, inputs_lengths, labels_length

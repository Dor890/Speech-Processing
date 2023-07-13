import torch
import librosa
import torchaudio
import numpy as np
import typing as tp
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.nn.functional import pairwise_distance
from scipy.spatial.distance import cdist
from fastdtw import fastdtw

from constants import N_MFCC, SR, HOP_LEN, N_FFT, N_MELS
MAX_LEN = 102400


def extract_features(wavs):
    """
    Extract MFCC features from the given audios batch.
    More ideas: try Time Domain / STFT / Mel Spectrogram
    """
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SR, n_mfcc=N_MFCC,
        melkwargs={'hop_length': HOP_LEN, 'n_fft': N_FFT, 'n_mels': N_MELS})
    mfcc_batch = mfcc_transform(wavs).squeeze()
    return mfcc_batch


class DTWModel:
    def __init__(self, x_train, y_train):
        self.x_train = extract_features(x_train)
        self.y_train = y_train

    def classify_using_DTW_distance(self, audio_files) -> tp.List[int]:
        """
        function to classify a given audio using DTW distance.
        audio_files: list of audio file paths or a a batch of audio files
         of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        predictions = []

        for wav in tqdm(audio_files):
            wav = torch.cat([wav, torch.zeros((1, MAX_LEN-wav.size(1)))], dim=1)
            best_dist, best_label = float('inf'), None
            mfcc = extract_features(wav)
            for i, x in enumerate(self.x_train):
                # cur_dist = self.DTW_distance(mfcc[0], x[0])
                cur_dist = fastdtw(mfcc, x)[0]
                if cur_dist < best_dist:
                    best_dist, best_label = cur_dist, self.y_train[i]
            predictions.append(best_label)

        return predictions

    def add_data(self, x, y):
        wavs = []
        for wav in x:
            wav = torch.cat([wav, torch.zeros((1, MAX_LEN-wav.size(1)))], dim=1)
            wavs.append(wav)
        wavs = torch.stack(wavs)
        self.x_train = torch.cat([self.x_train, extract_features(wavs)])
        self.y_train = self.y_train + y

    @staticmethod
    def DTW_distance(x, y):
        n, m = len(x), len(y)
        dtw_mat = np.zeros((n, m))
        dtw_mat[0, 0] = torch.sum(pairwise_distance(x[0], y[0], p=2))

        for i in range(1, n):
            dtw_mat[i, 0] = torch.sum(pairwise_distance(x[i], y[0], p=2))\
                            +dtw_mat[i-1, 0]

        for j in range(1, m):
            dtw_mat[0, j] = torch.sum(pairwise_distance(x[0], y[j], p=2))\
                            +dtw_mat[0, j-1]

        for i in range(1, n):
            for j in range(1, m):
                cost = torch.sum(pairwise_distance(x[i], y[j], p=2))
                dtw_mat[i, j] = cost+min(dtw_mat[i-1, j],
                                         dtw_mat[i, j-1],
                                         dtw_mat[i-1, j-1])

        return dtw_mat[n-1, m-1]


class EuclideanModel:
    def __init__(self, x_train, y_train):
        self.x_train = extract_features(x_train)
        self.y_train = y_train

    def classify_using_euclidean_distance(self, audio_files) -> tp.List[int]:
        """
        function to classify a given audio using euclidean distance.
        audio_files: list of audio file paths or a a batch of audio files
         of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        predictions = []

        for wav in tqdm(audio_files):
            wav = torch.cat([wav, torch.zeros((1, MAX_LEN-wav.size(1)))], dim=1)
            mfcc = extract_features(wav)
            best_dist, best_label = float('inf'), None
            for i, x in enumerate(self.x_train):
                cur_dist = torch.norm(mfcc - x)
                if cur_dist < best_dist:
                    best_dist, best_label = cur_dist, self.y_train[i]
            predictions.append(best_label)

        return predictions

    def add_data(self, x, y):
        wavs = []
        for wav in x:
            wav = torch.cat([wav, torch.zeros((1, MAX_LEN-wav.size(1)))], dim=1)
            wavs.append(wav)
        wavs = torch.stack(wavs)
        self.x_train = torch.cat([self.x_train, extract_features(wavs)])
        self.y_train = self.y_train + y

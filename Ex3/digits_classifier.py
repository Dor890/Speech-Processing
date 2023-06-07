import os
import torch
import librosa
import torchaudio
import typing as tp
import numpy as np
from fastdtw import fastdtw

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from word2number import w2n
from torch.nn.functional import pairwise_distance

TRAIN_PATH = "./train_files"
TEST_PATH = "./test_files"
OUTPUT_PATH = "output.txt"
CLASSIFIER_PATH = "digit_classifier.pt"

N_MFCC = 20


@dataclass
class ClassifierArgs:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with 
    default values (so run won't break when we test this).
    """
    # We will use this to give an absolute path to the data, make sure you
    # read the data using this argument.
    # You may assume the train data is the same
    path_to_training_data_dir: str = TRAIN_PATH
    path_to_testing_data_dir: str = TEST_PATH
    batch_size: int = 32
    num_epochs: int = 100
    sr = 16000

    # You may add other args here
    def __init__(self):
        x_train, y_train = self.load_train(self.path_to_training_data_dir)
        self.train_data = (x_train, y_train)

    @staticmethod
    def load_train(path: str) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        digit_directories = ["one", "two", "three", "four", "five"]
        audio_data, labels = [], []
        mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=N_MFCC)

        for digit_dir in digit_directories:
            digit_path = os.path.join(path, digit_dir)
            for filename in os.listdir(digit_path):
                if filename == ".DS_Store":
                    continue
                file_path = os.path.join(digit_path, filename)
                waveform, sample_rate = torchaudio.load(file_path)
                mfcc = mfcc_transform(waveform)
                audio_data.append(mfcc)
                labels.append(w2n.word_to_num(digit_dir))

        x_train = torch.stack(audio_data)
        y_train = torch.tensor(labels)

        return x_train, y_train

    @staticmethod
    def load_test(path: str):
        audio_data, labels = [], []

        for filename in os.listdir(path):
            if filename == ".DS_Store":
                continue
            file_path = os.path.join(path, filename)
            waveform, _ = torchaudio.load(file_path)
            audio_data.append(waveform)

        return torch.stack(audio_data)

    @staticmethod
    def load_test_from_list(files):
        audio_data, labels = [], []

        for filename in files:
            waveform, _ = torchaudio.load(filename)
            audio_data.append(waveform)

        return torch.stack(audio_data)


class DigitClassifier:
    """
    You should Implement your classifier object here
    """

    def __init__(self, args: ClassifierArgs):
        self.x_train, self.y_train = args.train_data

    @abstractmethod
    def classify_using_eucledian_distance(self, audio_files: tp.Union[
        tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using euclidean distance.
        audio_files: list of audio file paths or a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=N_MFCC)
        predictions = []

        if isinstance(audio_files, list):
            audio_files = ClassifierArgs.load_test_from_list(audio_files)

        for wave in audio_files:
            mfcc = mfcc_transform(wave)
            best_dist, best_label = float('inf'), None
            for i, x in enumerate(self.x_train):
                cur_dist = torch.sum(pairwise_distance(mfcc, x, p=2))
                if cur_dist < best_dist:
                    best_dist, best_label = cur_dist, self.y_train[i]
            predictions.append(best_label)

        return predictions

    @abstractmethod
    def classify_using_DTW_distance(self, audio_files: tp.Union[
        tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using DTW distance.
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=N_MFCC)
        predictions = []

        if isinstance(audio_files, list):
            audio_files = ClassifierArgs.load_test_from_list(audio_files)

        for wave in audio_files:
            dtw_mat = []
            mfcc = mfcc_transform(wave)
            for x in self.x_train:
                dtw_mat.append(self.DTW_distance(mfcc[0], x[0]))
            best_label = self.y_train[dtw_mat.index(min(dtw_mat))]
            predictions.append(best_label)

        return predictions

    @staticmethod
    def DTW_distance(x, y):
        n, m = len(x), len(y)
        dtw_mat = np.zeros((n, m))
        dtw_mat[0, 0] = torch.sum(pairwise_distance(x[0], y[0], p=2))

        for i in range(1, n):
            dtw_mat[i, 0] = torch.sum(pairwise_distance(x[i], y[0], p=2)) \
                            + dtw_mat[i - 1, 0]

        for j in range(1, m):
            dtw_mat[0, j] = torch.sum(pairwise_distance(x[0], y[j], p=2)) \
                            + dtw_mat[0, j - 1]

        for i in range(1, n):
            for j in range(1, m):
                cost = torch.sum(pairwise_distance(x[i], y[j], p=2))
                dtw_mat[i, j] = cost + min(dtw_mat[i - 1, j],
                                           dtw_mat[i, j - 1],
                                           dtw_mat[i - 1, j - 1])

        return dtw_mat[n - 1, m - 1]

    @abstractmethod
    def classify(self, audio_files: tp.List[str]) -> tp.List[str]:
        """
        function to classify a given audio using both distances.
        audio_files: list of ABSOLUTE audio file paths
        return: a list of strings of the following format: '{filename} - {predict using euclidean distance} - {predict using DTW distance}'
        Note: filename should not include parent path, but only the file name itself.
        """
        waves, predictions = [], []
        for file in audio_files:
            waveform, _ = torchaudio.load(file)
            waves.append(waveform)

        waves = torch.stack(waves)

        euc_predictions = self.classify_using_eucledian_distance(waves)
        dtw_predictions = self.classify_using_DTW_distance(waves)

        for i in range(len(audio_files)):
            if euc_predictions[i] != dtw_predictions[i]:
                print(f"diff at {i}, {audio_files[i]}")

            file_name = audio_files[i].split("\\")[1]
            predictions.append(f"{file_name} - {euc_predictions[i]} "
                               f"- {dtw_predictions[i]}")

        return predictions


class ClassifierHandler:

    @staticmethod
    def get_pretrained_model() -> DigitClassifier:
        """
        This function should load a pretrained / tuned 'DigitClassifier' object.
        We will use this object to evaluate your classifications.
        """
        args = ClassifierArgs()
        model = DigitClassifier(args)
        return model


def evaluate_model(model):
    """
    This function will be used to evaluate our model.
    The function will output file with the predictions of our model for the test set.
    """

    files = []
    for file_path in os.listdir(TEST_PATH):
        path = os.path.join(TEST_PATH, file_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")
        elif "DS_Store" in path:
            continue
        files.append(path)

    predictions = model.classify(files)
    with open(OUTPUT_PATH, "w") as file:
        file.writelines(line + '\n' for line in predictions)


if __name__ == '__main__':
    print(f"Start training on all data at {datetime.now()}")
    model = ClassifierHandler.get_pretrained_model()
    evaluate_model(model)
    print(f"Done training. Stop time:{datetime.now()}")
    # test_paths = [
    #     path
    #     for path in [
    #         os.path.join(TEST_PATH, name)
    #         for name in sorted(os.listdir(TEST_PATH))
    #     ]
    # ]
    # true_labels = [4, 2, 2, 1, 5, 4, 2, 1, 5, 4, 4, 2, 3, 2, 2, 1, 5, 4, 3, 4,
    #                4, 4, 4, 1, 2, 2, 2, 3, 2, 1, 3, 2, 4,
    #                1, 1, 1, 1, 5, 1, 2, 3, 2, 1, 5, 4, 5, 2, 3, 2, 4]
    # print('"' + '",\n "'.join(test_paths[len(true_labels):50]) + '"')
    # pred = model.classify_using_DTW_distance(test_paths[:len(true_labels)])
    # e_pred = model.classify_using_eucledian_distance(
    #     test_paths[:len(true_labels)])
    # for i in range(1, 6):
    #     m = sum([1 if t == i else 0 for t in true_labels])
    #     pred_acc = sum([1 if p == t and t == i else 0 for p, t in
    #                     zip(pred, true_labels)]) / m
    #     e_pred_acc = sum([1 if p == t and t == i else 0 for p, t in
    #                       zip(e_pred, true_labels)]) / m
    #     print(e_pred_acc, pred_acc)
    # print(sum([1 if p == t else 0 for p, t in zip(pred, true_labels)]) / len(
    #     true_labels))

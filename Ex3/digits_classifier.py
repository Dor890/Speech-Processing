import os
import torch
import librosa
import torchaudio
import typing as tp

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from word2number import w2n
from torch.nn.functional import pairwise_distance

TRAIN_PATH = "./train_files"
TEST_PATH = "./test_files"
OUTPUT_PATH = "output.txt"
CLASSIFIER_PATH = "digit_classifier.pt"

SR = 16000
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
        # self.path_to_testing_data_dir = self.load_test(self.path_to_testing_data_dir)
        self.train_data = (x_train, y_train)

    @staticmethod
    def load_train(path: str) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        digit_directories = ["one", "two", "three", "four", "five"]
        audio_data, labels = [], []
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=SR, n_mfcc=N_MFCC)

        for digit_dir in digit_directories:
            digit_path = os.path.join(path, digit_dir)
            for filename in os.listdir(digit_path):
                if filename == ".DS_Store":
                    continue
                file_path = os.path.join(digit_path, filename)
                waveform, sample_rate = torchaudio.load(file_path)
                mfcc = mfcc_transform(waveform)
                # mfcc = mfcc.flatten(start_dim=0)
                # mfcc = torch.nn.functional.normalize(mfcc, dim=0)
                audio_data.append(mfcc)
                labels.append(w2n.word_to_num(digit_dir))

        x_train = torch.stack(audio_data)
        y_train = torch.tensor(labels)

        return x_train, y_train

    @staticmethod
    def load_test(path: str):
        audio_data, labels = [], []

        for filename in os.listdir(path)[1:]:
            file_path = os.path.join(path, filename)
            waveform, sample_rate = torchaudio.load(file_path)
            audio_data.append(waveform)

        test_tensor = torch.stack(audio_data)

        return test_tensor


class DigitClassifier:
    """
    You should Implement your classifier object here
    """
    def __init__(self, args: ClassifierArgs):
        self.x_train, self.y_train = args.train_data

    @abstractmethod
    def classify_using_eucledian_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using euclidean distance.
        audio_files: list of audio file paths or a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        mfcc_transform = torchaudio.transforms.MFCC(sample_rate=SR,
                                                    n_mfcc=N_MFCC)
        predictions = []

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
    def classify_using_DTW_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using DTW distance.
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        mfcc_transform = torchaudio.transforms.MFCC(sample_rate=SR,
                                                    n_mfcc=N_MFCC)
        predictions = []

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
    def classify(self, audio_files: tp.List[str]) -> tp.List[str]:
        """
        function to classify a given audio using both distances.
        audio_files: list of ABSOLUTE audio file paths
        return: a list of strings of the following format: '{filename} - {predict using euclidean distance} - {predict using DTW distance}'
        Note: filename should not include parent path, but only the file name itself.
        """
        waves, files, predictions = [], [], []

        for file_path in os.listdir(audio_files)[1:]:
            path = os.path.join(audio_files, file_path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} does not exist")
            files.append(file_path)
            waveform, sr = torchaudio.load(path)
            waves.append(waveform)

        euc_predictions = self.classify_using_eucledian_distance(waves)
        dtw_predictions = self.classify_using_DTW_distance(waves)

        for i in range(len(files)):
            predictions.append(f"{files[i]} - {euc_predictions[i]} - {dtw_predictions[i]}")

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
    test_path = TEST_PATH
    predictions = model.classify(test_path)
    with open(OUTPUT_PATH, "w") as file:
        file.writelines(line+'\n' for line in predictions)


if __name__ == '__main__':
    print(f"Start training on all data at {datetime.now()}")
    model = ClassifierHandler.get_pretrained_model()
    evaluate_model(model)
    print(f"Done training. Stop time:{datetime.now()}")

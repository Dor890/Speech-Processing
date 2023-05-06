from abc import abstractmethod
import math
import torch
import torchaudio
from enum import Enum
import typing as tp
from dataclasses import dataclass
import json


class Genre(Enum):
    """
    This enum class is optional and defined for your convenience, you are not required to use it.
    Please use the int labels this enum defines for the corresponding genres in your predictions.
    """
    CLASSICAL: int = 0
    HEAVY_ROCK: int = 1
    REGGAE: int = 2


@dataclass
class TrainingParameters:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with 
    default values (so run won't break when we test this).
    """
    batch_size: int = 32
    num_epochs: int = 100
    train_json_path: str = "jsons/train.json"
    test_json_path: str = "jsons/test.json"

    def __init__(self, batch_size=32, num_epochs=100):
        self.batch_size, num_epochs = batch_size, num_epochs
        x_train, y_train = self.load_json(self.train_json_path)
        indices = torch.randperm(x_train.size()[0])
        x_train, y_train = x_train[indices], y_train[indices]
        x_train, y_train = torch.split(x_train, self.batch_size), \
                           torch.split(y_train, self.batch_size)  # 34 Batches
        self.train_data = (x_train, y_train)

        x_test, y_test = self.load_json(self.test_json_path)
        indices = torch.randperm(x_test.size()[0])
        x_test, y_test = x_test[indices], y_test[indices]
        self.test_data = (x_test, y_test)

    @staticmethod
    def load_json(path):
        mp3_arr, labels_arr = [], []
        with open(path, 'r') as f:
            data = json.load(f)
        for item in data:
            mp3_path = item['path']
            waveform, sr = torchaudio.load(mp3_path, format='mp3')
            mp3_arr.append(waveform.squeeze())
            label = item['label']
            if label == 'classical':
                labels_arr.append(Genre.CLASSICAL.value)
            elif label == 'heavy-rock':
                labels_arr.append(Genre.HEAVY_ROCK.value)
            elif label == 'reggae':
                labels_arr.append(Genre.REGGAE.value)
            else:
                raise RuntimeError("Unrecognized Label")
        mp3_tensor = torch.stack(mp3_arr)
        labels_tensor = torch.tensor(labels_arr)
        return mp3_tensor, labels_tensor


@dataclass
class OptimizationParameters:
    """
    This dataclass defines optimization related hyper-parameters to be passed to the model.
    feel free to add/change it as you see fit.
    """
    learning_rate: float = 0.01
    num_classes: int = 3
    input_dim: int = 66560
    sr: int = 16000
    n_fft: int = 1024
    hop_len: int = 512
    n_mels: int = 128


class MusicClassifier:
    """
    You should Implement your classifier object here
    """

    def __init__(self, opt_params: OptimizationParameters, **kwargs):
        """
        This defines the classifier object.
        - You should define your weights and biases as class components here.
        - You could use kwargs (dictionary) for any other variables you wish to pass in here.
        - You should use `opt_params` for your optimization and you are welcome to experiment.
        """
        self.weights = torch.zeros((opt_params.num_classes, opt_params.input_dim))
        self.biases = torch.zeros((opt_params.num_classes, ))
        self.opt_params = opt_params
        self.kwargs = kwargs

    def extract_feats(self, wavs: torch.Tensor):
        """
        this function extract features from a given audio.
        we will not be observing this method.
        """
        feats = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.opt_params.sr,
            n_fft=self.opt_params.n_fft,
            hop_length=self.opt_params.hop_len,
            n_mels=self.opt_params.n_mels)(wavs)

        feats = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(feats)

        # Reshape to (batch_size, n_feats)
        feats = torch.flatten(feats, start_dim=1)
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats

    def forward(self, feats: torch.Tensor) -> tp.Any:
        """
        this function performs a forward pass throuh the model, outputting scores for every class.
        feats: batch of extracted faetures
        """
        return torch.matmul(feats, self.weights.t()) + self.biases

    def backward(self, feats: torch.Tensor, output_scores: torch.Tensor, labels: torch.Tensor):
        """
        this function should perform a backward pass through the model.
        - calculate loss
        - calculate gradients
        - update gradients using SGD

        Note: in practice - the optimization process is usually external to the model.
        We thought it may result in less coding needed if you are to apply it here, hence 
        OptimizationParameters are passed to the initialization function
        """
        # Calculate loss
        batch_size = output_scores.size(0)
        loss = 0.0
        for i in range(batch_size):
            label_idx = labels[i].item()
            loss += -output_scores[i][label_idx] + math.log(torch.exp(output_scores[i]).sum())
        loss /= batch_size

        # Calculate gradients
        dW = torch.zeros_like(self.weights)
        db = torch.zeros_like(self.biases)
        for i in range(batch_size):
            label_idx = labels[i].item()
            exp_scores = torch.exp(output_scores[i])
            probs = exp_scores / exp_scores.sum()
            dW[label_idx] -= feats[i]
            dW += torch.outer(probs, feats[i])
            db[label_idx] -= 1
            db += probs

        # Update gradients using SGD
        self.weights = self.weights - self.opt_params.learning_rate * dW / batch_size
        self.biases = self.biases - self.opt_params.learning_rate * db / batch_size

        return loss

    def get_weights_and_biases(self):
        """
        This function returns the weights and biases associated with this model object, 
        should return a tuple: (weights, biases)
        """
        return (self.weights, self.biases)
    
    def classify(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        this method should receive a torch.Tensor of shape [batch, channels, time] (float tensor)
        and output a batch of corresponding labels [B, 1] (integer tensor)
        """
        feats = self.extract_feats(wavs)
        logits = self.forward(feats)
        preds = torch.argmax(logits, dim=1)
        return preds.unsqueeze(1)

    def set_weights(self, saved_weights):
        """
        This function should set the weights of the model to the given weights.
        """
        self.weights = saved_weights[0]
        self.biases = saved_weights[1]


class ClassifierHandler:

    @staticmethod
    def train_new_model(training_parameters: TrainingParameters) -> MusicClassifier:
        """
        This function should create a new 'MusicClassifier' object and train it from scratch.
        You could program your training loop / training manager as you see fit.
        """
        model = MusicClassifier(OptimizationParameters())

        for epoch in range(training_parameters.num_epochs):
            epoch_loss = 0.0
            for i in range(len(training_parameters.train_data[0])):
                batch_wavs, batch_labels = training_parameters.train_data[0][i],\
                                           training_parameters.train_data[1][i]
                feats = model.extract_feats(batch_wavs)
                logits = model.forward(feats)
                loss = model.backward(feats, logits, batch_labels)
                epoch_loss += loss

            print(f"Epoch {epoch}: Loss = {epoch_loss / len(training_parameters.train_data)}")

        model_dict = {"weights": model.weights, "biases": model.biases}
        torch.save(model_dict, 'model_files/music_classifier.pt')

        return model

    @staticmethod
    def test_music_classifier_accuracy(classifier: MusicClassifier,
                                       data: tp.Tuple) -> float:
        """
        This function should evaluate the accuracy of a trained
        'MusicClassifier' object on given data.
        It should return the accuracy as a float value between 0 and 1.
        """
        feats, labels = data
        preds = classifier.classify(feats)
        correct_preds = (preds == labels).sum().item()
        total_preds = len(labels)
        accuracy = correct_preds / total_preds
        return accuracy

    @staticmethod
    def get_pretrained_model() -> MusicClassifier:
        """
        This function should construct a 'MusicClassifier' object, load it's
         trained weights / hyperparameters and return the loaded model.
        """
        model = MusicClassifier(OptimizationParameters())
        loaded_dict = torch.load("model_files/music_classifier.pt")
        saved_weights = (loaded_dict["weights"], loaded_dict["biases"])
        model.set_weights(saved_weights)
        return model


def main():
    train_params = TrainingParameters()
    ClassifierHandler.train_new_model(train_params)
    model = ClassifierHandler.get_pretrained_model()
    accuracy = ClassifierHandler.test_music_classifier_accuracy(model, train_params.test_data)
    print(accuracy)


if __name__ == '__main__':
    main()
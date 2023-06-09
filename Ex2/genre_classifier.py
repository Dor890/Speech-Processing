import librosa
import torch
import torchaudio
import librosa.feature as lib
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
    num_epochs: int = 1
    train_json_path: str = "jsons/train.json"
    test_json_path: str = "jsons/test.json"


@dataclass
class OptimizationParameters:
    """
    This dataclass defines optimization related hyper-parameters to be passed to the model.
    feel free to add/change it as you see fit.
    """
    learning_rate: float = 0.05
    num_classes: int = 3
    input_dim: int = 76960
    sr: int = 22050
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
        self.weights = torch.randn(
            (opt_params.num_classes, opt_params.input_dim))
        self.biases = torch.zeros((opt_params.num_classes,))
        self.opt_params = opt_params
        self.kwargs = kwargs

    def extract_feats(self, wavs: torch.Tensor):
        """
        this function extract features from a given audio.
        we will not be observing this method.
        """
        feats = torch.tensor([])
        for wav in wavs:
            # extraction pipeline
            mel_Spec = self.extract_mel_spec(wav)
            sc = self.extract_spectral_centroid(wav)
            mfcc = self.extract_MFCC(wav)
            wav_numpy = wav.numpy()
            zrc = self.extract_zrc(wav_numpy)
            rms = self.extract_rms(wav_numpy)
            spec_cons = self.extract_spectral_contrast(wav_numpy)

            # Stack features by row
            feature = torch.hstack(
                (mel_Spec, sc, mfcc, zrc, rms, spec_cons)).flatten()

            # make sure inp
            if feature.shape[0] < self.opt_params.input_dim:
                pad_length = self.opt_params.input_dim - feature.shape[0]
                feature = torch.nn.functional.pad(feature, (0, pad_length),
                                                  value=0)

            elif feature.shape[0] > self.opt_params.input_dim:
                feature = feature[:self.opt_params.input_dim]

            # Concat to matrix
            feats = torch.cat((feats, feature))
        return feats.reshape(len(wavs), -1)

    # --- Specific feature extraction methods ---

    def extract_mel_spec(self, wav):
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.opt_params.sr,
            n_fft=self.opt_params.n_fft,
            hop_length=self.opt_params.hop_len,
            n_mels=self.opt_params.n_mels)
        mel_spec = mel_spec_transform(wav)
        mel_spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_spec)
        mel_spec = mel_spec.flatten(start_dim=0)
        mel_spec = torch.nn.functional.normalize(mel_spec, dim=0)
        return mel_spec

    def extract_MFCC(self, wav):
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.opt_params.sr,
            n_mfcc=10,
            melkwargs={'hop_length': self.opt_params.hop_len,
                       'n_fft': self.opt_params.n_fft})
        mfccs = mfcc_transform(wav)
        mfccs = mfccs.flatten(start_dim=0)
        return torch.nn.functional.normalize(mfccs, dim=0)

    def extract_spectral_centroid(self, wav):
        sc_transform = torchaudio.transforms.SpectralCentroid(
            sample_rate=self.opt_params.sr,
            hop_length=self.opt_params.hop_len,
            n_fft=self.opt_params.n_fft)
        centroid = sc_transform(wav)
        centroid = centroid.flatten(start_dim=0)
        return torch.nn.functional.normalize(centroid, dim=0)

    def extract_zrc(self, wav):
        zrc = lib.zero_crossing_rate(
            y=wav, hop_length=self.opt_params.hop_len).flatten()
        to_torch = torch.from_numpy(zrc).float()
        return torch.nn.functional.normalize(to_torch, dim=0)

    def extract_rms(self, wav):
        rms = lib.rms(y=wav, hop_length=self.opt_params.hop_len).flatten()
        to_torch = torch.from_numpy(rms).float()
        return torch.nn.functional.normalize(to_torch, dim=0)

    def extract_spectral_contrast(self, wav):
        spec_constrast = lib.spectral_contrast(
            y=wav, sr=self.opt_params.sr, hop_length=self.opt_params.hop_len,
            n_fft=self.opt_params.n_fft).flatten()
        to_torch = torch.from_numpy(spec_constrast).float()
        return torch.nn.functional.normalize(to_torch, dim=0)

    # --- End of feature extraction methods ---

    def forward(self, feats: torch.Tensor) -> tp.Any:
        """
        this function performs a forward pass through the model,
        outputting scores for every class.
        feats: batch of extracted features.
        """
        x = feats @ self.weights.T + self.biases
        return torch.sigmoid(x)

    def backward(self, feats: torch.Tensor, output_scores: torch.Tensor,
                 labels: torch.Tensor):
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
        loss = -(labels * torch.log(output_scores) +
                 (1 - labels) * torch.log(1 - output_scores)).mean()

        # Calculate gradients
        dL_dy = output_scores - labels
        dL_dw = feats.T @ dL_dy
        dL_db = torch.sum(dL_dy, dim=0)

        # Update gradients using SGD
        self.weights -= self.opt_params.learning_rate * dL_dw.T
        self.biases -= self.opt_params.learning_rate * dL_db

        return loss

    def get_weights_and_biases(self):
        """
        This function returns the weights and biases associated with this model object,
        should return a tuple: (weights, biases)
        """
        return self.weights, self.biases

    def classify(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        this method should receive a torch.Tensor of shape [batch, channels, time] (float tensor)
        and output a batch of corresponding labels [B, 1] (integer tensor)
        """
        feats = self.extract_feats(wavs)
        logits = self.forward(feats)
        preds = torch.argmax(logits, dim=1).unsqueeze(1)
        return preds

    def set_weights(self, saved_weights):
        """
        This function should set the weights of the model to the given weights.
        """
        self.weights = saved_weights[0]
        self.biases = saved_weights[1]


class ClassifierHandler:
    # --- data loading pipeline ---
    @staticmethod
    def load_train_data(training_parameters, all_data=False):
        x_train, y_train = ClassifierHandler.load_json(
            training_parameters.train_json_path)
        x_test, y_test = ClassifierHandler.load_json(
            training_parameters.test_json_path)

        if all_data:
            # Train on all the data provided
            waves = torch.cat((x_train, x_test))
            all_test = torch.cat((y_train, y_test))
            indices = torch.randperm(waves.size()[0])
            waves = waves[indices]
            all_test = all_test[indices]

            waves, all_test = torch.split(waves,
                                          training_parameters.batch_size), \
                              torch.split(all_test,
                                          training_parameters.batch_size)  # 32 Batches

            return waves, all_test

        else:
            # Split for model evaluation
            indices = torch.randperm(x_train.size()[0])
            x_train, y_train = x_train[indices], y_train[indices]
            x_train, y_train = torch.split(x_train,
                                           training_parameters.batch_size), \
                               torch.split(y_train,
                                           training_parameters.batch_size)  # 32 Batches

            indices = torch.randperm(x_test.size()[0])
            x_test, y_test = x_test[indices], y_test[indices]

            return x_train, y_train, x_test, y_test

    @staticmethod
    def load_json(path):
        mp3_arr, labels_arr = [], []
        with open(path, 'r') as f:
            data = json.load(f)
        for item in data:
            mp3_path = item['path']

            # load and convert to torch
            waveform, sr = librosa.load(mp3_path,
                                        sr=OptimizationParameters.sr)
            waveform = torch.from_numpy(waveform)

            mp3_arr.append(waveform.squeeze())
            ClassifierHandler.parse_label(item, labels_arr)

        mp3_tensor = torch.stack(mp3_arr)
        labels_tensor = torch.tensor(labels_arr)
        return mp3_tensor, labels_tensor

    @staticmethod
    def parse_label(item, labels_arr):
        label = item['label']
        if label == 'classical':
            labels_arr.append(Genre.CLASSICAL.value)
        elif label == 'heavy-rock':
            labels_arr.append(Genre.HEAVY_ROCK.value)
        elif label == 'reggae':
            labels_arr.append(Genre.REGGAE.value)
        else:
            raise RuntimeError("Unrecognized Label")

    @staticmethod
    def convert_labels_tensor(labels):
        labels = labels.reshape((len(labels), 1))
        new_labels = torch.zeros((len(labels), 3))
        new_labels.scatter_(1, labels, 1)
        return new_labels

    # --- end of pipe ---

    @staticmethod
    def train_new_model(
            training_parameters: TrainingParameters) -> MusicClassifier:
        """
        This function should create a new 'MusicClassifier' object and train
        it from scratch.
        You could program your training loop / training manager as you see fit.
        """
        model = MusicClassifier(OptimizationParameters())
        x_train, y_train = ClassifierHandler.load_train_data(
            training_parameters)[:2]

        for epoch in range(training_parameters.num_epochs):
            for i in range(len(x_train)):  # Batch
                batch_wavs, batch_labels = x_train[i], y_train[i]
                feats = model.extract_feats(batch_wavs)
                output_scores = model.forward(feats)
                model.backward(feats, output_scores,
                               ClassifierHandler.convert_labels_tensor(
                                   batch_labels))

        torch.save(model.weights,
                   'model_files/music_classifier_new_weights.pt')
        torch.save(model.biases,
                   'model_files/music_classifier_new_biases.pt')

        return model

    @staticmethod
    def get_pretrained_model() -> MusicClassifier:
        """
        This function should construct a 'MusicClassifier' object, load it's
         trained weights / hyperparameters and return the loaded model.
        """
        model = MusicClassifier(OptimizationParameters())
        a = torch.load("model_files/trained_weights.pt")
        b = torch.load("model_files/trained_biases.pt")
        model.set_weights((a, b))
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

        # normalize labels to match classify api
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)

        correct_preds = (preds == labels).sum().item()
        total_preds = len(labels)
        accuracy = correct_preds / total_preds
        return accuracy


def main():
    train_params = TrainingParameters()
    # ClassifierHandler.train_new_model(train_params)
    model = ClassifierHandler.get_pretrained_model()
    accuracy = ClassifierHandler. \
        test_music_classifier_accuracy(model,
                                       ClassifierHandler.load_train_data(
                                           train_params)[2:])
    print(accuracy)


if __name__ == '__main__':
    main()

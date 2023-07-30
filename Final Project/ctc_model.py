import torch
import torchaudio

import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder._ctc_decoder import download_pretrained_files

from tqdm import tqdm

from TextTransform import gd
from constants import SR, HOP_LEN, N_FFT, N_MELS, N_MFCC, HIDDEN_DIM, \
    NUM_LAYERS, \
    NUM_CLASSES, N_EPOCHS, BATCH_SIZE, WEIGHT_DECAY, LEARNING_RATE, \
    CTC_MODEL_PATH, DROPOUT
from decoders import GreedyDecoder, BeamSearchDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
files = download_pretrained_files("librispeech-4-gram")


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout,
                 n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride,
                              padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride,
                              padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    """Speech Recognition Model Inspired by DeepSpeech 2"""

    def __init__(self, vocabulary, n_cnn_layers, n_rnn_layers, rnn_dim,
                 n_class, n_feats,
                 stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        self.vocabulary = vocabulary
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride,
                             padding=3 // 2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout,
                        n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

        # Decoders
        self.greedy_decoder = GreedyDecoder(vocabulary.translator.values())
        # self.beam_decoder = ctc_decoder(lexicon='lexicon.txt',
        #                                 tokens='tokens.txt', lm=None)

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x


def save_model(model, path):
    """
    Saves a pytorch models to the given path.
    """
    torch.save(model.state_dict(), '{}'.format(path))


def load_model(model, path):
    """
    Loads a pytorch models from the given path. The models should already by
    created (e.g. by calling the constructor) and should be passed as an argument.
    """
    model.load_state_dict(torch.load('{}'.format(path)))
    model.eval()


def extract_features(wavs, is_train=False):
    """
    Extract MFCC features from the given audios batch.
    More ideas: try Time Domain / STFT / Mel Spectrogram
    """
    spectrograms, input_lengths = [], []

    # MFCC Transform
    # transform = torchaudio.transforms.MFCC(
    #     sample_rate=SR, n_mfcc=N_MFCC)
    # mfcc_batch = mfcc_transform(wavs).squeeze()
    # mfcc_batch = mfcc_batch.permute(0, 2, 1)
    # return mfcc_batch

    # transform = torchaudio.transforms.MelSpectrogram(SR)
    # mel_batch = transform(wavs).squeeze()
    # mel_batch = mel_batch.permute(0, 2, 1)
    # return mel_batch
    if is_train:
        transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_mels=N_MELS),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35))
    else:
        transform = torchaudio.transforms.MelSpectrogram()

    for wav in wavs:
        spec = transform(wav).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        input_lengths.append(spec.shape[0] // 2)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)

    # mel_batch = mel_batch.permute(0, 2, 1)  # (batch, mel, timeFrame)
    return spectrograms, torch.Tensor(input_lengths).long()


class LSTMModel(nn.Module):
    """
    A basic LSTM models for speech recognition.
    """

    def __init__(self, vocabulary, lang_model=None):
        super(LSTMModel, self).__init__()
        self.vocabulary = vocabulary
        self.lang_model = lang_model

        # RNN layers
        self.rnn = nn.LSTM(input_size=4032, hidden_size=HIDDEN_DIM,
                           num_layers=NUM_LAYERS, batch_first=True)

        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)

        # Fully connected layer
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

        # Decoders
        self.greedy_decoder = GreedyDecoder(vocabulary.translator.values())
        # self.beam_decoder = ctc_decoder(lexicon='lexicon.txt',
        #                                 tokens='tokens.txt', lm=files.lm)

    def forward(self, x):
        x = self.conv(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        rnn_output, _ = self.rnn(x)
        output = self.fc(rnn_output)
        # output = F.log_softmax(output, dim=2)

        return output


def predict(model, feats):
    """
    Predicts a batch of waveforms using the given models.
    """
    emission = model(feats)
    greedy_result = model.greedy_decoder(emission)
    # beam_search_result = model.beam_decoder(emission)
    return greedy_result


def test(model, test_loader, criterion):
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)
            # arg_maxes = torch.argmax(output.transpose(0, 1), dim=2)
            decoded_preds, decoded_targets = gd(output.transpose(0, 1), labels, label_lengths)
            print(decoded_preds, decoded_targets)


def train_all_data(model, train_loader, criterion):
    data_len = len(train_loader.dataset)
    optimizer = torch.optim.RMSprop(model.parameters(), LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                                              steps_per_epoch=data_len,
                                              epochs=N_EPOCHS,
                                              anneal_strategy='linear')
    model.train()
    model = model.to(device)
    for epoch in range(N_EPOCHS):
        if (epoch + 1) % 11 == 0:
            save_model(model, CTC_MODEL_PATH)

        for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            scheduler.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(spectrograms),
                                                                           data_len,
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

    save_model(model, CTC_MODEL_PATH)

import torch
import torchaudio

import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder._ctc_decoder import download_pretrained_files

from tqdm import tqdm

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
        return self.layer_norm(x)

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
                             hidden_size=rnn_dim, dropout=dropout,
                             batch_first=i == 0)
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
        self.beam_decoder = ctc_decoder(lexicon='lexicon.txt',
                                        tokens='tokens.txt', lm=None)


    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[3],
                   sizes[2])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x


def save_model(model, path):
    """
    Saves a pytorch model to the given path.
    """
    torch.save(model.state_dict(), '{}'.format(path))


def load_model(model, path):
    """
    Loads a pytorch model from the given path. The model should already by
    created (e.g. by calling the constructor) and should be passed as an argument.
    """
    model.load_state_dict(torch.load('{}'.format(path)))
    model.eval()


def extract_features(wavs):
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

    transform = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_mels=N_MELS),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35)
    )


    for wav in wavs:
        spectrograms.append(transform(wav))
        input_lengths.append(spectrograms[-1].shape[2])

    max_length = max(tensor.size(2) for tensor in spectrograms)
    # Pad tensors and create the big tensor
    mel_batch = torch.zeros((len(spectrograms), N_MELS, max_length))
    for i, tensor in enumerate(spectrograms):
        mel_batch[i, :, :tensor.shape[2]] = tensor[0, :, :]
        # debug
        for j in range(tensor.shape[1]):
            for k in range(tensor.shape[2]):
                assert mel_batch[i][j][k] == tensor[0][j][k]
                # print(mel_batch[i][j][k],tensor[0][j][k])

    mel_batch = mel_batch.permute(0, 2, 1) # (batch, mel, timeFrame)
    return mel_batch, torch.Tensor(input_lengths).long()


class LSTMModel(nn.Module):
    """
    A basic LSTM model for speech recognition.
    """

    def __init__(self, vocabulary, lang_model=None):
        super(LSTMModel, self).__init__()
        self.vocabulary = vocabulary
        self.lang_model = lang_model

        # RNN layers
        self.rnn = nn.LSTM(input_size=N_MELS, hidden_size=HIDDEN_DIM,
                           num_layers=NUM_LAYERS, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

        # Decoders
        self.greedy_decoder = GreedyDecoder(vocabulary.translator.values())
        self.beam_decoder = ctc_decoder(lexicon='lexicon.txt',
                                        tokens='tokens.txt', lm=files.lm)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        output = self.fc(rnn_output)
        # output = F.log_softmax(output, dim=2)

        return output

def predict(model, feats):
    """
    Predicts a batch of waveforms using the given model.
    """
    emission = model(feats)
    # greedy_result = model.gredy_decoder(emission)
    beam_search_result = model.beam_decoder(emission)
    return beam_search_result


def targets_to_tensor(vocabulary, targets):
    """
    Converts a list of targets to a tensor.
    """
    max_length = max(len(target) for target in targets)
    translated_tensors, lengths = [], []

    for target in targets:
        translated_numbers = [vocabulary.translator[char] for char in target]
        lengths.append(len(translated_numbers))
        padded_numbers = F.pad(torch.tensor(translated_numbers),
                               (0, max_length - len(translated_numbers)))
        translated_tensors.append(padded_numbers)
    final_tensor = torch.stack(translated_tensors)
    lengths = torch.tensor(lengths)

    # test reconstruction
    j = 0
    for tensor in final_tensor:
        t = ""
        for i in range(tensor.shape[0]):
            t += vocabulary.invert_trans[tensor[i].item()]
        assert t == targets[j]
        j += 1

    return final_tensor, lengths


def train_batch(model, optimizer, feats_batch, target_batch, input_lengths,
                scheduler):
    """
    Trains a single batch of the model, using the given optimizer.
    """
    feats_batch.to(device)
    target_batch, targets_lengths = targets_to_tensor(model.vocabulary,
                                                      target_batch)
    target_batch.to(device)

    optimizer.zero_grad()

    # Forward pass
    output = model(feats_batch).to(device)
    output = F.log_softmax(output, dim=2).permute(1, 0, 2)

    # target_batch = torch.flatten(target_batch)
    # target_batch = target_batch[target_batch != 0]
    # Calculate the loss after CTC and perform autograd
    loss = F.ctc_loss(output, target_batch, input_lengths, targets_lengths,
                      zero_infinity=True).to(device)

    # Backward pass and optimize
    loss.backward()
    # torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    return loss.item()


def train_all_data(model, train_data, target_data):
    optimizer = torch.optim.AdamW(model.parameters(), 0.003,
                                  weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                                              steps_per_epoch=16,
                                              epochs=N_EPOCHS,
                                              anneal_strategy='linear')
    model.to(device)
    for epoch in range(N_EPOCHS):
        total_loss = 0
        model.train()
        for i, batch_start in tqdm(
                enumerate(range(0, len(train_data), BATCH_SIZE))):
            batch = train_data[batch_start:batch_start + BATCH_SIZE]

            feats_batch, input_lengths = extract_features(batch) # we premute (batch, mel, frame(numMel))
            target_batch = target_data[batch_start:batch_start + BATCH_SIZE]
            loss = train_batch(model, optimizer, feats_batch, target_batch,
                               input_lengths, scheduler)
            total_loss += loss

        print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}")

    save_model(model, CTC_MODEL_PATH)


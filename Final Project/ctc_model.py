import torch
import torchaudio

import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder

from tqdm import tqdm

from constants import SR, HOP_LEN, N_FFT, N_MELS, N_MFCC, HIDDEN_DIM, NUM_LAYERS,\
    NUM_CLASSES, N_EPOCHS, BATCH_SIZE, WEIGHT_DECAY, LEARNING_RATE, CTC_MODEL_PATH, DROPOUT
from decoders import GreedyDecoder, BeamSearchDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    #     sample_rate=SR, n_mfcc=N_MFCC,
    #     melkwargs={'hop_length': HOP_LEN, 'n_fft': N_FFT, 'n_mels': N_MELS})
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
        input_lengths.append(spectrograms[-1].shape[2] // 2)
    max_length = max(tensor.size(2) for tensor in spectrograms)
    # Pad tensors and create the big tensor
    mel_batch = torch.zeros((len(spectrograms), N_MELS, max_length))
    for i, tensor in enumerate(spectrograms):
        mel_batch[i, :, :tensor.shape[2]] = tensor[0, :, :]
    mel_batch = mel_batch.permute(0, 2, 1)
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
                                        tokens='tokens.txt', lm=None)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        output = self.fc(rnn_output)

        output = F.log_softmax(output, dim=2)

        return output


class BidirectionalGRU(nn.Module):
    def __init__(self, vocabulary, batch_first=True):
        super(BidirectionalGRU, self).__init__()
        self.vocabulary = vocabulary
        self.BiGRU = nn.GRU(
            input_size=N_MELS, hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(N_MELS)
        self.dropout = nn.Dropout(DROPOUT)

        # Decoders
        self.greedy_decoder = GreedyDecoder(vocabulary.translator.values())
        self.beam_decoder = ctc_decoder(lexicon='lexicon.txt',
                                        tokens='tokens.txt', lm=None)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        x = F.log_softmax(x, dim=2)
        return x


def predict(model, feats):
    """
    Predicts a batch of waveforms using the given model.
    """
    emission = model(feats)
    greedy_result = model.greedy_decoder(emission)
    # beam_search_result = self.beam_decoder(emission)
    return greedy_result


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
                               (0, max_length-len(translated_numbers)))
        translated_tensors.append(padded_numbers)
    final_tensor = torch.stack(translated_tensors)
    lengths = torch.tensor(lengths)
    return final_tensor, lengths


def train_batch(model, optimizer, feats_batch, target_batch, input_lengths, scheduler):
    """
    Trains a single batch of the model, using the given optimizer.
    """
    feats_batch.to(device)
    target_batch, targets_lengths = targets_to_tensor(model.vocabulary, target_batch)
    target_batch.to(device)

    optimizer.zero_grad()

    # Forward pass
    output = model(feats_batch).permute(1, 0, 2).to(device)

    # Calculate the loss after CTC and perform autograd
    loss = F.ctc_loss(output, target_batch, input_lengths, targets_lengths).to(device)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()


def train_all_data(model, train_data, target_data):
    optimizer = torch.optim.AdamW(model.parameters(), LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                                            steps_per_epoch=int(len(train_data)//BATCH_SIZE-1),
                                            epochs=N_EPOCHS,
                                            anneal_strategy='linear')
    model.train()
    model.to(device)
    for epoch in range(N_EPOCHS):
        total_loss = 0
        for i, batch_start in tqdm(enumerate(range(0, len(train_data), BATCH_SIZE))):
            batch = train_data[batch_start:batch_start+BATCH_SIZE]
            feats_batch, input_lengths = extract_features(batch)
            target_batch = target_data[batch_start:batch_start+BATCH_SIZE]
            loss = train_batch(model, optimizer, feats_batch, target_batch, input_lengths, scheduler)
            total_loss += loss

        print(f"Epoch: {epoch+1}, Loss: {total_loss:.4f}")

    save_model(model, CTC_MODEL_PATH)

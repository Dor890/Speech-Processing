import torch
import torchaudio

import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder

from tqdm import tqdm

from constants import SR, HOP_LEN, N_FFT, N_MELS, N_MFCC, HIDDEN_SIZE, NUM_LAYERS,\
    NUM_CLASSES, N_EPOCHS, BATCH_SIZE, WEIGHT_DECAY, LEARNING_RATE, TIME, MODEL_PATH
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


def extract_features(wavs, permute=True):
    """
    Extract MFCC features from the given audios batch.
    More ideas: try Time Domain / STFT / Mel Spectrogram
    """
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SR, n_mfcc=N_MFCC,
        melkwargs={'hop_length': HOP_LEN, 'n_fft': N_FFT, 'n_mels': N_MELS})
    mfcc_batch = mfcc_transform(wavs).squeeze()
    if permute:
        mfcc_batch = mfcc_batch.permute(0, 2, 1)
    return mfcc_batch

    # transform = torchaudio.transforms.MelSpectrogram(SR)
    # mel_batch = transform(wavs).squeeze()
    # if permute:
    #     mel_batch = mel_batch.permute(0, 2, 1)
    # return mel_batch


class LSTMModel(nn.Module):
    """
    A basic LSTM model for speech recognition.
    """
    def __init__(self, vocabulary):
        super(LSTMModel, self).__init__()
        self.vocabulary = vocabulary

        # RNN layers
        self.rnn = nn.LSTM(input_size=N_MFCC, hidden_size=HIDDEN_SIZE,
                           num_layers=NUM_LAYERS, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

        # Decoders
        self.greedy_decoder = GreedyDecoder(vocabulary.translator.values())
        self.beam_decoder = ctc_decoder(lexicon='lexicon.txt',
                                        tokens='tokens.txt', lm=None)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        output = self.fc(rnn_output)

        output = F.log_softmax(output, dim=2)

        return output

    def predict(self, x):
        """
        Predicts a batch of waveforms.
        """
        feats = extract_features(x)
        emission = self(feats)
        greedy_result = self.greedy_decoder(emission)
        # beam_search_result = self.beam_decoder(emission)
        return greedy_result

    def targets_to_tensor(self, targets):
        """
        Converts a list of targets to a tensor.
        """
        max_length = max(len(target) for target in targets)
        translated_tensors, lengths = [], []

        for target in targets:
            translated_numbers = [self.vocabulary.translator[char] for char in target]
            lengths.append(len(translated_numbers))
            padded_numbers = F.pad(torch.tensor(translated_numbers),
                                   (0, max_length-len(translated_numbers)))
            translated_tensors.append(padded_numbers)
        final_tensor = torch.stack(translated_tensors)
        lengths = torch.tensor(lengths)
        return final_tensor, lengths


def train_batch(model, optimizer, mfcc_batch, target_batch):
    """
    Trains a single batch of the model, using the given optimizer.
    """
    optimizer.zero_grad()

    # Forward pass
    output = model(mfcc_batch).permute(1, 0, 2).to(device)

    # Calculate the loss after CTC and perform autograd
    target_batch, targets_lengths = model.targets_to_tensor(target_batch)
    length_batch = torch.full(size=(target_batch.shape[0],), fill_value=TIME)
    loss = F.ctc_loss(output, target_batch, length_batch, targets_lengths).to(device)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    return loss.item()


def train_all_data(model, train_data, target_data):
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    model.train()

    for epoch in range(N_EPOCHS):
        total_loss = 0
        for i, batch_start in tqdm(enumerate(range(0, len(train_data), BATCH_SIZE))):
            batch = train_data[batch_start:batch_start+BATCH_SIZE]
            mfcc_batch = extract_features(batch)
            target_batch = target_data[batch_start:batch_start+BATCH_SIZE]
            loss = train_batch(model, optimizer, mfcc_batch, target_batch)
            total_loss += loss

        print(f"Epoch: {epoch+1}, Loss: {total_loss:.4f}")

    save_model(model, MODEL_PATH)

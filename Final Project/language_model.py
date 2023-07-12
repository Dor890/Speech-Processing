import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from constants import N_EPOCHS, LANG_MODEL_PATH, BATCH_SIZE, LEARNING_RATE,\
    HIDDEN_DIM, PAD_TOKEN, EMBED_DIM, SEQ_LEN
from ctc_model import save_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LanguageModel(nn.Module):
    """
    A basic LSTM model for speech recognition.
    """
    def __init__(self, vocabulary):
        super(LanguageModel, self).__init__()
        self.vocabulary = vocabulary
        self.embedding = nn.Embedding(vocabulary.size, EMBED_DIM)
        self.rnn = nn.GRU(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, vocabulary.size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden


def tokenize(sentence, vocabulary):
    """
    Tokenize the sentence into a list of words, represented by numbers.
    """
    translator = vocabulary.translator
    new_sent = []
    for word in sentence:
        new_sent.append(translator[word])
    return new_sent


def prepare_batch(sentences, vocabulary):
    """
    Prepare the batch for the model, by tokenizing each sentence and
    converting to tensors.
    """
    tokenized_batch = [tokenize(sentence, vocabulary) for sentence in sentences]

    lengths = [len(seq) for seq in tokenized_batch]
    max_length = max(lengths)

    padded_batch = [torch.LongTensor(pad_sequence(seq, max_length)) for seq in tokenized_batch]

    data = torch.stack(padded_batch)
    lengths_tensor = torch.LongTensor(lengths)

    return data, lengths_tensor


def pad_sequence(sequence, max_length):
    """
    Pad the sequence with the PAD_TOKEN to the max_length.
    """
    if len(sequence) < max_length:
        padded_sequence = sequence + [PAD_TOKEN] * (max_length - len(sequence))
    else:
        padded_sequence = sequence[:max_length]
    return padded_sequence


def train_sequence(model, inputs, targets, lengths, criterion, optimizer):
    """
    Train the model on a single sequence.
    """
    model.train()
    hidden = None
    optimizer.zero_grad()

    outputs, hidden = model(inputs, hidden)

    sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
    sorted_outputs = outputs[sorted_indices]
    packed_outputs = nn.utils.rnn.pack_padded_sequence(sorted_outputs, sorted_lengths, batch_first=True)
    sorted_targets = targets[sorted_indices]
    packed_targets = nn.utils.rnn.pack_padded_sequence(sorted_targets, sorted_lengths, batch_first=True)

    loss = criterion(packed_outputs.data, packed_targets.data)
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    return loss.item()


def train_all_data(model, train_data):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.to(device)
    model.train()
    total_loss = 0
    n_tokens = 0

    for epoch in tqdm(range(N_EPOCHS)):
        for i in range(0, len(train_data), BATCH_SIZE):
            batch_data = train_data[i:i+BATCH_SIZE]
            data, lengths = prepare_batch(batch_data, model.vocabulary)
            data = data.to(device)
            lengths = lengths.to(device)
            model.train()

            for j in range(0, len(data)-SEQ_LEN, SEQ_LEN):
                inputs = torch.tensor(data[:, j:j+SEQ_LEN]).to(device)
                targets = torch.tensor(data[:, j+1:j+SEQ_LEN+1]).to(device)

                loss = train_sequence(model, inputs, targets, lengths, criterion, optimizer)
                total_loss += loss
                n_tokens += SEQ_LEN

            if (i+1) % 200 == 0:
                avg_loss = total_loss / n_tokens
                print(f"Epoch [{epoch+1}/{N_EPOCHS}], Step [{i+1}/{len(train_data)}], Loss: {avg_loss:.4f}")

    save_model(model, LANG_MODEL_PATH)

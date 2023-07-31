import os
import torch
import random
import matplotlib.pyplot as plt
from jiwer import wer, cer
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import ctc_model
import language_model
from data import Data, AN4Dataset, data_processing
from distances import DTWModel, EuclideanModel
from vocabulary import Vocabulary
from constants import BATCH_SIZE, SR, CTC_MODEL_PATH, LEARNING_RATE, N_EPOCHS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hparams = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 5,
    "rnn_dim": 512,
    "n_class": 29,
    "n_feats": 128,
    "stride": 2,
    "dropout": 0.1,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": N_EPOCHS
}


def evaluate(model, x_test, y_test):
    """
    Evaluate the models over the test set.
    """
    predictions, targets = [], []

    for i, batch_start in tqdm(enumerate(range(0, len(x_test), BATCH_SIZE))):
        batch = x_test[batch_start:batch_start + BATCH_SIZE]
        # feats = torch.zeros((len(batch), 1, MAX_LEN))
        # for i, tensor in enumerate(batch):
        #     padded_tensor = torch.cat(
        #         [tensor, torch.zeros((1, MAX_LEN-tensor.size(1)))], dim=1)
        #     big_tensor[i] = padded_tensor
        feats, _ = ctc_model.extract_features(batch)
        batch_preds = ctc_model.predict(model, feats)
        for j in range(len(batch_preds)):
            pred_tokens = model.beam_decoder.idxs_to_tokens(batch_preds[j][0].tokens)
            if j % 50 == 0:
                print(f'True transcription: {y_test[batch_start + j]}')
                print(f'Predicted transcription: {pred_tokens}')
            predictions.append(pred_tokens)
            targets.append(y_test[batch_start + j])
            # plot_alignments(batch[j],
            #                 models(models.extract_features(batch[j])),
            #                 pred_tokens, batch_preds[j].timesteps)

    wer_error = wer(targets, predictions)
    cer_error = cer(targets, predictions)
    return wer_error, cer_error


def plot_alignments(waveform, emission, tokens, timesteps):
    """
    Plots the alignment between the waveform and the predicted transcription.
    """
    fig, ax = plt.subplots(figsize=(32, 10))
    ax.plot(waveform)

    ratio = waveform.shape[0] / emission.shape[1]
    word_start = 0
    for i in range(len(tokens)):
        if i != 0 and tokens[i - 1] == "|":
            word_start = timesteps[i]
        if tokens[i] != "|":
            plt.annotate(tokens[i].upper(), (timesteps[i] * ratio, waveform.max() * 1.02), size=14)
        elif i != 0:
            word_end = timesteps[i]
            ax.axvspan(word_start * ratio, word_end * ratio, alpha=0.1, color="red")

    xticks = ax.get_xticks()
    plt.xticks(xticks, xticks / SR)
    ax.set_xlabel("Time")
    ax.set_xlim(0, waveform.shape[0])
    plt.show()


def test_distance_algorithms(data):
    """
    Test the Distances algorithms (DTW & Euclidean) as the most naive
    implementations.
    """
    x_train, y_train = data.get_data('train')
    x_val, y_val = data.get_data('val')
    x_test, y_test = data.get_data('test')

    dtw = DTWModel(x_train, y_train)
    dtw.add_data(x_val, y_val)
    predictions_dtw = dtw.classify_using_DTW_distance(x_test)
    print('Predictions:')
    print(predictions_dtw[:5])
    print('True labels:')
    print(y_test[:5])
    print('Testing DTW algorithm...')
    wer_error = wer(y_test, predictions_dtw)
    cer_error = cer(y_test, predictions_dtw)
    print(f'DTW Test WER: {wer_error:.4f}')
    print(f'DTW Test CER: {cer_error:.4f}')
    print('DTW tested successfully')

    # print('Testing Euclidean algorithm...')
    # euclidean = EuclideanModel(x_train, y_train)
    # euclidean.add_data(x_val, y_val)
    # predictions_euclidean = euclidean.classify_using_euclidean_distance(x_test)
    # wer_error = wer(y_test, predictions_euclidean)
    # cer_error = cer(y_test, predictions_euclidean)
    # print('Predictions:')
    # print(predictions_euclidean[:5])
    # print('True labels:')
    # print(y_test[:5])
    # print(f'Euclidean Test WER: {wer_error:.4f}')
    # print(f'Euclidean Test CER: {cer_error:.4f}')
    # print('Euclidean tested successfully')


def main():
    print('--- Start running ---')
    data = Data()

    # test_distance_algorithms(data)
    x_val, y_val = data.get_data('val')
    x_train, y_train = data.get_data('train')

    train_data_set = AN4Dataset('train')
    test_data_set = AN4Dataset('test')
    train_loader = DataLoader(dataset=train_data_set,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              collate_fn=lambda x: data_processing(x, vocabulary, 'train'))
    test_loader = DataLoader(dataset=test_data_set,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             collate_fn=lambda x: data_processing(x, vocabulary, 'val'))

    vocabulary = Vocabulary(transcriptions=(y_train + y_val))
    # lang_model = language_model.LanguageModel(vocabulary)
    print('Training the language models...')
    # language_model.train_all_data(lang_model, y_train+y_val)
    print('Language models trained successfully')
    ctc_lstm = ctc_model.SpeechRecognitionModel(vocabulary,
                                                hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
                                                hparams['n_class'], hparams['n_feats'], hparams['stride'],
                                                hparams['dropout']
                                                ).to(device)
    #
    # ctc_lstm = ctc_model.LSTMModel(vocabulary)
    lossFunc = nn.CTCLoss(blank=28, zero_infinity=True).to(device)
    ctc_model.load_model(ctc_lstm, CTC_MODEL_PATH)
    # ctc_model.train_all_data(ctc_lstm, train_loader, lossFunc)

    # if os.path.exists(CTC_MODEL_PATH):
    #     print('Loading the models...')
    #     ctc_model.load_model(ctc_lstm, CTC_MODEL_PATH)
    #     print('Model loaded successfully')
    # else:  # Train the models
    #     print('Training the models...')
    #     ctc_model.train_all_data(ctc_lstm, train_loader, lossFunc)
    print('Model trained successfully')

    ctc_model.test(ctc_lstm, test_loader, lossFunc)

    # Evaluate the models on the test set
    # print('Evaluating the models...')
    # data = Data()
    # x_test, y_test = data.get_data('train')
    # test_wer, test_cer = evaluate(ctc_lstm, x_test, y_test)
    # print(f'Test WER: {test_wer:.4f}')
    # print(f'Test CER: {test_cer:.4f}')
    # print('Model evaluated successfully')
    print('-- Finished running ---')


if __name__ == '__main__':
    main()

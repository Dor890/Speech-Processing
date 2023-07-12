import os
import torch
import random
import matplotlib.pyplot as plt
from jiwer import wer, cer
from tqdm import tqdm

import model
from data import Data
from distances import DTWModel, EuclideanModel
from vocabulary import Vocabulary
from constants import BATCH_SIZE, MAX_LEN, SR, MODEL_PATH


def evaluate(model, x_test, y_test):
    """
    Evaluate the model over the test set.
    """
    predictions, targets = [], []

    for i, batch_start in tqdm(enumerate(range(0, len(x_test), BATCH_SIZE))):
        batch = x_test[batch_start:batch_start+BATCH_SIZE]
        big_tensor = torch.zeros((len(batch), 1, MAX_LEN))
        for i, tensor in enumerate(batch):
            padded_tensor = torch.cat(
                [tensor, torch.zeros((1, MAX_LEN-tensor.size(1)))], dim=1)
            big_tensor[i] = padded_tensor
        batch_preds = model.predict(big_tensor)
        for j in batch_preds:
            pred_tokens = model.beam_decoder.idxs_to_tokens(batch_preds[j].tokens)
            if j % 50 == 0:
                print(f'True transcription: {y_test[batch_start+j]}')
                print(f'Predicted transcription: {pred_tokens}')
            predictions.append(pred_tokens)
            targets.append(y_test[batch_start+j])
            plot_alignments(batch[j],
                            model(model.extract_features(batch[j])),
                            pred_tokens, batch_preds[j].timesteps)

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
    vocabulary = Vocabulary(transcriptions=(y_train+y_val))
    LSTM_model = model.LSTMModel(vocabulary)
    if os.path.exists(MODEL_PATH):
        print('Loading the model...')
        model.load_model(LSTM_model, MODEL_PATH)
        print('Model loaded successfully')
    else:  # Train the model
        print('Training the model...')
        model.train_all_data(LSTM_model, x_train, y_train)
    print('Model trained successfully')

    # Evaluate the model on the test set
    print('Evaluating the model...')
    x_test, y_test = data.get_data('test')
    test_wer, test_cer = evaluate(LSTM_model, x_train, y_train)
    print(f'Test WER: {test_wer:.4f}')
    print(f'Test CER: {test_cer:.4f}')
    print('Model evaluated successfully')
    print('-- Finished running ---')


if __name__ == '__main__':
    main()

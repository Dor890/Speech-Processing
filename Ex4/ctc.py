import numpy as np
import sys

BLANK = 'ϵ'


def ctc(y, label, tokens):
    T, K = y.shape
    L = len(label)
    blank_index = tokens.index(BLANK)
    # Initialize z
    z = [BLANK]
    for i in range(L):
        z.append(label[i])
        z.append(BLANK)

    # Initialize alpha
    alpha = np.zeros((len(z), T))
    alpha[0][0] = y[0][blank_index]  # Probability of blank to be first token
    alpha[1][0] = y[0][tokens.index(label[0])]  # Probability of first label to be first token
    # alpha[s][0] = 0 for every s > 2
    # Dynamic programming for calculating all alpha matrix
    for t in range(1, T):
        alpha[0][t] = alpha[0][t-1] * y[t][blank_index]
        alpha[1][t] = (alpha[0][t-1] + alpha[1][t-1]) * y[t][tokens.index(label[0])]
        for s in range(2, len(z)):  # len(z) = 2L + 1
            zs_idx = tokens.index(label[s])
            cur_y = y[t][zs_idx]
            if z[s] == BLANK or (s >= 2 and z[s] == z[s-2]):
                alpha[s][t] += (alpha[s-1][t-1] + alpha[s][t-1]) * cur_y
            else:
                alpha[s][t] += (alpha[s-2][t-1] + alpha[s-1][t-1] + alpha[s][t-1]) * cur_y

    # Return the probability of the label
    p = alpha[2*L-1][T-1] + alpha[2*L][T-1]
    return p


def print_p(p: float):
    print("%.3f" % p)


if __name__ == "__main__":
    # Load the network outputs from the provided numpy file
    y_path = sys.argv[1]
    y = np.load(y_path)  # shape: (T, K), T=time steps, K=number of tokens

    # Get the label and tokens from the command line arguments
    label = sys.argv[2]
    tokens = sys.argv[3]

    # Calculate the probability P(p|x)
    p = ctc(y, label, tokens)

    # Print the result
    print_p(p)

import numpy as np
import sys

BLANK = 'ϵ'
BLACK_INDEX = 0


def ctc(y, label, tokens):
    T, K = y.shape
    L, M = len(label), len(tokens)
    if T < L or K != M+1:
        return 0  # y shape is not valid


    # Initialize z
    z = [BLANK]
    for i in range(L):
        z.append(label[i])
        z.append(BLANK)

    # Initialize alpha
    alpha = np.zeros((len(z), T))
    alpha[0][0] = y[0][BLACK_INDEX]  # Probability of blank to be first token
    alpha[1][0] = y[0][tokens.index(label[0])+1]  # Probability of first label to be first token
    # alpha[s][0] = 0 for every s > 2

    # Dynamic programming for calculating all alpha matrix
    for t in range(1, T):
        alpha[0][t] = alpha[0][t-1] * y[t][BLACK_INDEX]
        alpha[1][t] = (alpha[0][t-1] + alpha[1][t-1]) * y[t][tokens.index(label[0])+1]
        for s in range(2, len(z)):  # len(z) = 2L + 1
            zs_idx = tokens.index(z[s])+1 if z[s] in tokens else BLACK_INDEX
            cur_y = y[t][zs_idx]
            if z[s] == BLANK or z[s] == z[s-2]:
                alpha[s][t] = (alpha[s-1][t-1] + alpha[s][t-1]) * cur_y
            else:
                alpha[s][t] = (alpha[s-2][t-1] + alpha[s-1][t-1] + alpha[s][t-1]) * cur_y

    # Return the probability of the label
    return alpha


def prob_from_alpha(alpha):
    return alpha[-1][-1] + alpha[-2][-1]


def print_p(p: float):
    print("%.3f" % p)


def create_test():
    y = np.array([
        [0.1, 0.2, 0.3, 0.4],  # Time step 1
        [0.2, 0.1, 0.4, 0.3],  # Time step 2
        [0.3, 0.4, 0.1, 0.2],  # Time step 3
        [0.4, 0.3, 0.2, 0.1],  # Time step 4
        [0.1, 0.2, 0.3, 0.4],  # Time step 5
    ])

    np.save('test_mat.npy', y)


def main():
    # Load the network outputs from the provided numpy file
    y_path = sys.argv[1]
    y = np.load(y_path)  # shape: (T, K), T=time steps, K=number of tokens

    # Get the label and tokens from the command line arguments
    label = sys.argv[2]
    tokens = sys.argv[3]
    if not all(letter in tokens for letter in label):
        print_p(0)
        return

    # Calculate the probability P(p|x)
    alpha = ctc(y, label, tokens)
    p = prob_from_alpha(alpha)  # Last two cells in the last column

    # Print the result
    print_p(p)


if __name__ == "__main__":
    # create_test()
    main()

# Numerical constants
SR = 16000
FILE_2CHECK = 13
HOP_LEN = 160
N_FFT = 400
N_MELS = 128  # 128 for Mel_Spec, 23 for MFCC
N_MFCC = 13
N_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0001
NUM_LAYERS = 3
HIDDEN_DIM = 64
EMBED_DIM = 300
NUM_CLASSES = 28
TIME = 513  # 513 for Mel_Spec, 641 for MFCC
PAD_TOKEN = 0
SEQ_LEN = 3
DROPOUT = 0.1

# Strings constants
CTC_MODEL_PATH = 'models/ctc_model.pth'
LANG_MODEL_PATH = 'models/lang_model.pth'

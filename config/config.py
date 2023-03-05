import string

alphabet = string.ascii_lowercase + string.digits + string.punctuation
NUM_CHAR_TOKENS = len(alphabet) +2

output_seq_char_len = 290
LEN_CHAR_VOCAB = 28
LINE_DEPTH = 15
TOTAL_LINE_DEPTH = 20
CLASSES = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
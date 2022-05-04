import transformers
import numpy as np

max_len = 128
batch_size = 32
EPOCHS = 10

ROBERTA_PATH = 'xlm-roberta-base'
Training_file = 'data/train.csv'
Testing_file = 'data/test.csv'
model_load_path = 'xnli_best_model.bin'
model_save_path = 'models/zeroshot_english_test2.bin'
XNLI_file = 'train.csv'
target_language = 'english'
mode = 'train'

tokenizer = transformers.AutoTokenizer.from_pretrained(ROBERTA_PATH, do_lower_case = True)

DEVICE = 'cuda'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
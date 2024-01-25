from pathlib import Path
import numpy as np
import mlx.core as mx

text_data_dir = Path(__file__).parent.parent.parent / 'data/text'
class TextData:
    def __init__(self, config, data_file_name):
        # config
        self.train_validation_split = config['train_validation_split']
        self.block_size = config['block_size']
        self.batch_size = config['batch_size']
        # saving other values
        self.data_file_name = data_file_name
        # initialising required values
        self.text = 'No text loaded'
        self.vocab = []
        self.string_to_int = {}
        self.int_to_string = {}
        # loading the dataset completely
        self._load()

    def __str__(self):
        return self.text[:100]

    def _load(self):
        with open(text_data_dir / self.data_file_name, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.vocab = sorted(list(set(self.text)))
        self.string_to_int = {char:index for index, char in enumerate(self.vocab)}
        self.int_to_string = {index:char for index, char in enumerate(self.vocab)}
        self.train_set, self.validation_set = self._get_train_validation_splits()

    def _get_train_validation_splits(self):
        full_data = self.encode(self.text)
        n = int(self.train_validation_split*len(full_data))
        return mx.array(full_data[:n]), mx.array(full_data[n:])

    def get_random_batch(self, split):
        data = self.train_set if split == 'train' else self.validation_set
        sentence_starting_points = np.asarray(mx.random.randint(
                low=0,
                high=len(data)-self.block_size,
                shape=(self.batch_size, )
            ))
        xs = mx.stack([data[int(i): int(i)+self.block_size] for i in sentence_starting_points])
        ys = mx.stack([data[int(i)+1: int(i)+1+self.block_size] for i in sentence_starting_points])
        return xs, ys


    def encode(self, input_text):
        encoder = lambda x: [self.string_to_int[char] for char in x]
        return encoder(input_text)

    def decode(self, input_vector):
        decoder = lambda x: [self.int_to_string[value] for value in x]
        return decoder(input_vector)



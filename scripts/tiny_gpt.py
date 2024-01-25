import sys
sys.path.append('/Users/m.omar/Desktop/GPT_from_scratch')
from config import Config
from _src.dataloader import TextData
import mlx.core as mx


mx.random.seed(0)
text_data = TextData(Config().data.text_data, 'shakespeare.txt')



from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

import torch
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import ListField, TextField
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
    ELMoTokenCharactersIndexer,
    PretrainedTransformerIndexer,
    PretrainedTransformerMismatchedIndexer,
)
from allennlp.data.tokenizers import (
    CharacterTokenizer,
    PretrainedTransformerTokenizer,
    SpacyTokenizer,
    WhitespaceTokenizer,
)
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import (
    Embedding,
    TokenCharactersEncoder,
    ElmoTokenEmbedder,
    PretrainedTransformerEmbedder,
    PretrainedTransformerMismatchedEmbedder,
)
from allennlp.nn import util as nn_util

import warnings
warnings.filterwarnings("ignore")



@DatasetReader.register('classification-pt-tsv')
class ClassificationPtTsvReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 model_name: str = 'bert-base-cased',
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name=model_name,
                                                                     max_length=max_tokens)
        self.token_indexers = token_indexers or {'tokens': PretrainedTransformerIndexer(model_name=model_name,
                                                                                        max_length=max_tokens)}
        self.max_tokens = max_tokens

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r', encoding="latin-1") as lines:
            for line in lines:
                text, sentiment, sentence_id = line.strip().split('\t')
                sentiment = sentiment.split(',')[0]
                yield self.text_to_instance(text, sentiment)


# reader = ClassificationPtTsvReader()
# dataset = reader.read('data/original/train.tsv')
#
# print('type of dataset: ', type(dataset))
# print('type of its first element: ', type(dataset[0]))
# print('size of dataset: ', len(dataset))
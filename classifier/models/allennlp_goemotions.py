from typing import Dict

import torch
import numpy
from allennlp.data import DataLoader, Instance, Token, Vocabulary
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data.fields import TextField, LabelField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import Model

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import (
    Embedding,
    TokenCharactersEncoder,
    ElmoTokenEmbedder,
    PretrainedTransformerEmbedder,
    PretrainedTransformerMismatchedEmbedder,
)

from allennlp.nn import util
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlp.training.metrics.fbeta_measure import FBetaMeasure

@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.macrof1 = FBetaMeasure(average='macro')
        self.microf1 = FBetaMeasure(average='micro')
        self.weightedf1 = FBetaMeasure(average='weighted')


    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        self.accuracy(logits, label)
        self.macrof1(logits, label)
        self.microf1(logits, label)
        self.weightedf1(logits, label)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset),
                "macrof1_precision": self.macrof1.get_metric(reset)["precision"],
                "macrof1_recall": self.macrof1.get_metric(reset)["recall"],
                "macrof1_fscore": self.macrof1.get_metric(reset)["fscore"],
                "microf1_precision": self.microf1.get_metric(reset)["precision"],
                "microf1_recall": self.microf1.get_metric(reset)["recall"],
                "microf1_fscore": self.microf1.get_metric(reset)["fscore"],
                "weightedf1_precision": self.weightedf1.get_metric(reset)["precision"],
                "weightedf1_recall": self.weightedf1.get_metric(reset)["recall"],
                "weightedf1_fscore": self.weightedf1.get_metric(reset)["fscore"]
                }

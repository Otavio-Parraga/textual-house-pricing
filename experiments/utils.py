from flair.data import Sentence
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, TransformerWordEmbeddings
from tqdm import tqdm
import pandas as pd
from pathlib import Path


# relevant features
relevant_features = ['area',
                     'rooms',
                     'bathrooms',
                     'suites',
                     'parking_slots',
                     'price']


# metrics
def median_absolute_percentage_error(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true))


class LoadPreComputedEmbeddings(BaseEstimator, TransformerMixin):
    def __init__(self,
                 data_source='rent',
                 embeddings='word',
                 description='description_preprocess',
                 fine_tune=False):
        self.data_source = data_source
        self.embeddings = embeddings
        self.description = description
        self.fine_tune = fine_tune
        #self.data_path = f"../datasets/embeddings/{self.embeddings}/{self.data_source}_{self.description}_emb.csv"
        self.data_path = self.__build_embedding_path(self.data_source, self.embeddings, self.description, self.fine_tune)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.read_csv(self.data_path, index_col=False)
        return df.loc[X.index].to_numpy()

    def __build_embedding_path(self, data_source,embeddings,description,fine_tune):
        base_path = Path('../datasets/embeddings/fine-tuned') if fine_tune else Path('../datasets/embeddings') 
        return base_path / f'{embeddings}' / f'{data_source}_{description}_emb.csv'


# These snippets are repeated from generate_embeddings
class DocumentEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.doc_embeddings = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        embedding_features = []
        for sentence in tqdm(X, desc='Embedding Sentences'):
            if (sentence == ' ') or (len(sentence) < 1):
                embedding_features.append(np.zeros((300,)))
            else:
                sentence = Sentence(sentence)
                self.doc_embeddings.embed(sentence)
                embedding_features.append(sentence.embedding.cpu().numpy())
        return np.array(embedding_features)


class BERTTransformer(DocumentEmbeddingTransformer):
    def __init__(self):
        super().__init__()
        self.bert = TransformerWordEmbeddings('neuralmind/bert-base-portuguese-cased', model_max_length=512,
                                              cache_dir=f"./pretrained_save/'neuralmind/bert-base-portuguese-cased")
        self.bert.max_subtokens_sequence_length = 512
        self.doc_embeddings = DocumentPoolEmbeddings([self.bert])


class WordEmbeddingsTransformer(DocumentEmbeddingTransformer):
    def __init__(self):
        super().__init__()
        self.doc_embeddings = DocumentPoolEmbeddings(
            [WordEmbeddings('pt')])
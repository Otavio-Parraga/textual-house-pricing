from flair.data import Sentence
import pandas as pd
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, TransformerWordEmbeddings
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path

# transformer class
class DocumentEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.doc_embeddings = None
        self.embedding_size = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        embedding_features = []
        for sentence in tqdm(X, desc='Embedding Sentences'):
            if (sentence == ' ') or (len(sentence) < 1):
                embedding_features.append(np.zeros((self.embedding_size,)))
            else:
                sentence = Sentence(sentence)
                self.doc_embeddings.embed(sentence)
                embedding_features.append(sentence.embedding.detach().cpu().numpy())
        return np.array(embedding_features)


class BERTTransformer(DocumentEmbeddingTransformer):
    def __init__(self, lang='pt', fine_tune=False, dataset=None):
        super().__init__()
        if lang == 'pt':
            base_model = 'neuralmind/bert-base-portuguese-cased' if not fine_tune else f'fine-tuned-bert-{dataset}'
            print(f'Using {base_model}')
            self.bert = TransformerWordEmbeddings(base_model, model_max_length=512,
                                              cache_dir=f"./pretrained_save/'neuralmind/bert-base-portuguese-cased")
        elif lang == 'en':
            base_model = 'bert-base-uncased' if not fine_tune else f'fine-tuned-bert-{dataset}'
            print(f'Using {base_model}')
            self.bert = TransformerWordEmbeddings(base_model, model_max_length=512,
                                              cache_dir=f"./pretrained_save/'bert-base-uncased")
        self.bert.max_subtokens_sequence_length = 512
        self.doc_embeddings = DocumentPoolEmbeddings([self.bert])
        self.embedding_size = 768


class WordEmbeddingsTransformer(DocumentEmbeddingTransformer):
    def __init__(self, lang='pt', fine_tune=False):
        super().__init__()
        if lang == 'pt':
            self.doc_embeddings = DocumentPoolEmbeddings(
                [WordEmbeddings('pt', fine_tune=fine_tune, force_cpu=False)])
        elif lang == 'en':
            self.doc_embeddings = DocumentPoolEmbeddings(
                [WordEmbeddings('en', fine_tune=fine_tune, force_cpu=False)])
        self.embedding_size = 300

def filter_dataset(df):
    df = df[df['type'].str.contains('Apartamento')]
    # select only those with descriptions
    df = df[df['description'] != ' ']
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', nargs='+', default=['rent', 'sale', 'homes'], required=True)
    parser.add_argument('-emb', '--embeddings', nargs='+', default=['word', 'bert'], required=True)
    parser.add_argument('-ft', '--fine_tune', action='store_true')
    args = parser.parse_args()

    sales = pd.read_json('../datasets/preprocessed/poa-sale.json')
    rents = pd.read_json('../datasets/preprocessed/poa-rent.json')
    homes = pd.read_json('../datasets/preprocessed/homes.json')

    save_path = Path("../datasets/embeddings/fine-tuned") if args.fine_tune else Path("../datasets/embeddings")
    save_path.mkdir(parents=True, exist_ok=True)

    Path(save_path / 'word').mkdir(parents=True, exist_ok=True)
    Path(save_path / 'bert').mkdir(parents=True, exist_ok=True)

    descriptions = ['description_preprocess', 'description_no_stp']

    # rents
    if 'rent' in args.dataset and 'word' in args.embeddings:
        print('Generating embeddings rent-word')
        embedding = WordEmbeddingsTransformer(fine_tune=args.fine_tune)
        for desc in descriptions:
            embedded_sentences = embedding.transform(rents[desc])
            temp_df = pd.DataFrame(embedded_sentences)
            temp_df.to_csv(save_path / "word" / f"rent_{desc}_emb.csv", index=False)

    # sales
    if 'sale' in args.dataset and 'word' in args.embeddings:
        print('Generating embeddings sale-word')
        embedding = WordEmbeddingsTransformer(fine_tune=args.fine_tune)
        for desc in descriptions:
            embedded_sentences = embedding.transform(sales[desc])
            temp_df = pd.DataFrame(embedded_sentences)
            temp_df.to_csv(save_path / "word" / f"sale_{desc}_emb.csv", index=False)

    # homes
    if 'homes' in args.dataset and 'word' in args.embeddings:
        print('Generating embeddings homes-word')
        embedding = WordEmbeddingsTransformer(lang='en', fine_tune=args.fine_tune)
        for desc in descriptions:
            embedded_sentences = embedding.transform(homes[desc])
            temp_df = pd.DataFrame(embedded_sentences)
            temp_df.to_csv(save_path / "word" / f"homes_{desc}_emb.csv", index=False)
    
    # rents
    if 'rent' in args.dataset and 'bert' in args.embeddings:
        print('Generating embeddings rent-bert')
        embedding = BERTTransformer(fine_tune=args.fine_tune, dataset='rent')
        for desc in descriptions:
            embedded_sentences = embedding.transform(rents[desc])
            temp_df = pd.DataFrame(embedded_sentences)
            temp_df.to_csv(save_path / "bert" / f"rent_{desc}_emb.csv", index=False)

    # sales
    if 'sale' in args.dataset and 'bert' in args.embeddings:
        print('Generating embeddings sale-bert')
        embedding = BERTTransformer(fine_tune=args.fine_tune, dataset='sale')
        for desc in descriptions:
            embedded_sentences = embedding.transform(sales[desc])
            temp_df = pd.DataFrame(embedded_sentences)
            temp_df.to_csv(save_path / "bert" / f"sale_{desc}_emb.csv", index=False)


    if 'homes' in args.dataset and 'bert' in args.embeddings:
        print('Generating embeddings homes-bert')
        embedding = BERTTransformer(lang='en', fine_tune=args.fine_tune, dataset='homes')
        for desc in descriptions:
            embedded_sentences = embedding.transform(homes[desc])
            temp_df = pd.DataFrame(embedded_sentences)
            temp_df.to_csv(save_path / "bert" / f"homes_{desc}_emb.csv", index=False)
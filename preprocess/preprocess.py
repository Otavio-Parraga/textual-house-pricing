from nltk.corpus import stopwords
import os
import pandas as pd
import pathlib
import re
import unidecode


def split_punctuation(content):
    content = re.findall(r"[\w']+|[-.,!?;""]", str(content))
    return ' '.join(content)

def split_numbers_and_letters(string):
    return ' '.join(re.split('(\d+)',string))

def lower(string):
    return str(string).lower()

def remove_stps(text, stopwords_language):
    text = text.split()
    new_text = []
    for char in text:
        if char not in stopwords.words(stopwords_language):
            new_text.append(char)
    return ' '.join(new_text)

def normalize_str(string):
    return unidecode.unidecode(string)

def full_cleaning(string, remove_stp=True, stopwords_language='portuguese'):
    string = lower(string)
    string = normalize_str(string)
    string = split_punctuation(string)
    string = split_numbers_and_letters(string)
    if remove_stp: string = remove_stps(string, stopwords_language=stopwords_language)
    return string



if __name__ == '__main__':
    datasets_dir = pathlib.Path(__file__).parent.parent.resolve().joinpath('datasets')
    originals_dir = datasets_dir.joinpath('original')
    preprocessed_dir = datasets_dir.joinpath('preprocessed')
    originals = os.listdir(originals_dir)

    for dataset in originals:
        df = pd.read_json(f'{originals_dir}/{dataset}')
        stopwords_language = 'english' if dataset == 'homes.json' else 'portuguese'

        print(f"{dataset} stopwords language is {stopwords_language}")

        df['description_no_stp'] = df['description'].apply(lambda x: full_cleaning(x, stopwords_language=stopwords_language))
        df['description_preprocess'] = df['description'].apply(lambda x: full_cleaning(x, False))

        df = df.drop(
            columns=[
                'images',
                'amenities',
                'url',
                'street_number',
                'street_name',
                'neighborhood',
            ],
            errors='ignore',
        )

        df.to_json(f'{preprocessed_dir}/{dataset}', orient='records')

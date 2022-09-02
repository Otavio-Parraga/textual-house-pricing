# TEXTUAL HOUSE PRICING (BRACIS 2022)

## Virtual environment creation and dependencies install

```
python -m venv .env
. .env/Scripts/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

***
# Running experiments:
* Run preprocess script on original data:
  ```
  cd preprocess
  python preprocess.py
  ```
* To run the baseline models:
  * Available datasets are: *rent* and *sale*
  ```
  cd baseline
  python baseline.py -d dataset
  ```
* To run the experiments:
  ```
  cd experiments
  python experiment_name.py -d dataset
  ```
* In preprocessing we store all code to create the embeddings and preprocess the datasets.
* In Interpretability our code to create permutation analysis based on the models.
* In transformers we have the code used to train the BERT model, further used to generate contextual embeddings.
***


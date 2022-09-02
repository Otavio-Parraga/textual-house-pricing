# https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
import pandas as pd
from transformers import AutoTokenizer, BertForMaskedLM, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import random
import os
import numpy as np
from argparse import ArgumentParser
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)
os.environ['PYHTONHASHSEED'] = str(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True


parser = ArgumentParser()
parser.add_argument('-d', '--dataset', choices=['rent', 'sale', 'homes'], required=True)
parser.add_argument('-e', '--epochs', type=int, default=2)

args = parser.parse_args()

output_folder = Path(f'./fine-tuned-bert-{args.dataset}')
output_folder.mkdir(exist_ok=True, parents=True)

if args.dataset in ('rent', 'sale'):
    df = pd.read_json(f'../datasets/preprocessed/poa-{args.dataset}.json')
    # filters
    df = df[df['type'].str.contains('Apartamento')]
    # select only those with descriptions
    df = df[df['description'] != ' ']
    df = df['description']

    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', cache_dir=f'./pretrained_save/')
    model = BertForMaskedLM.from_pretrained('neuralmind/bert-base-portuguese-cased', cache_dir=f'./pretrained_save/')
else:
    df = pd.read_json(f'../datasets/preprocessed/{args.dataset}.json')
    df = df[df['type'] != 'lots land']
    df = df.dropna()
    df = df['description']
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', cache_dir=f'./pretrained_save/')
    model = BertForMaskedLM.from_pretrained('bert-base-cased', cache_dir=f'./pretrained_save/')



list_of_texts = [i for i in df]

inputs = tokenizer(list_of_texts, return_tensors='pt', max_length=128, truncation=True, padding='max_length')

inputs['labels'] = inputs.input_ids.detach().clone()

# create random array of floats with equal dimensions to input_ids tensor
rand = torch.rand(inputs.input_ids.shape)
# create mask array
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
           (inputs.input_ids != 102) * (inputs.input_ids != 0)

selection = []

for i in range(inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )

for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = 103

class MeditationsDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

dataset = MeditationsDataset(inputs)

loader = DataLoader(dataset, batch_size=6, shuffle=True)

model.to(device)
model.train()
optim = AdamW(model.parameters(), lr=5e-5)

epochs = args.epochs

for epoch in range(epochs):
    loop = tqdm(loader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

    model.save_pretrained(output_folder)
    tokenizer.save_pretrained(output_folder)


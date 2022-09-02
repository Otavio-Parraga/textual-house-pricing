import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import RegressorTransformer
from dataset import RealEstateDataset, filter_json, read_json_data
from sklearn.model_selection import KFold
from train import train
from evaluation import evaluate, average_metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=3, type=int)
    parser.add_argument('-bsz', '--batch_size', default=32)
    parser.add_argument('-pt', '--pretrained',
                        default='neuralmind/bert-base-portuguese-cased')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['rent', 'sale'], required=True, help='available datasets are rent and sale')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_PATH = f'../datasets/preprocessed/poa-{args.dataset}.json'
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    instances = filter_json(read_json_data(DATA_PATH))

    results = []

    for split_id, (train_index, test_index) in enumerate(kf.split(instances)):
        model = RegressorTransformer(
            args.pretrained, cache_dir=f'./pretrained_save/{args.pretrained}')

        train_dataset = RealEstateDataset(
            DATA_PATH, model.tokenizer, indexes=train_index, description_option='description_preprocess')
        test_dataset = RealEstateDataset(
            DATA_PATH, model.tokenizer, indexes=test_index, description_option='description_preprocess')

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
        criterion = nn.MSELoss()
        #criterion = nn.L1Loss()

        criterion.to(device)
        model.to(device)

        print(f'Split {split_id+1}')
        train(model, train_loader, test_loader,
              optimizer, criterion, device, args.epochs)
        results.append(evaluate(model, test_loader, device))
    print(average_metrics(results))

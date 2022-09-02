# TODO: evaluation method
# TODO: Collect metrics, like other models
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from utils import median_absolute_percentage_error, rmse

metrics_dict = {'MAE': mean_absolute_error, 'MeanAPE': mean_absolute_percentage_error,
                'MSE': mean_squared_error, 'R2': r2_score, 'MedianAPE': median_absolute_percentage_error, 'RMSE': rmse}


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    ground_truth = []
    predictions = []
    for text, price in tqdm(dataloader, total=len(dataloader), desc='Running Evaluation'):

        input_ids = text['input_ids'].to(device)
        attention_mask = text['attention_mask'].to(device)
        price = price.to(device)

        output = model(input_ids, attention_mask)

        ground_truth.append(price.squeeze())
        predictions.append(output.squeeze())

    ground_truth = np.exp(torch.cat(ground_truth).cpu().numpy())
    predictions = np.exp(torch.cat(predictions).cpu().numpy())

    return {i: m(ground_truth, predictions) for i, m in metrics_dict.items()}


def average_metrics(metrics):
    metrics_averaged = {}
    for k, _ in metrics_dict.items():
        metrics_averaged[k] = average_metric(metrics, k)
    return metrics_averaged


def average_metric(metrics, name):
    values = [items[name] for items in metrics]
    return np.mean(values)

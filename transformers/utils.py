from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import mean_squared_error

def get_scheduler(dataloader, epochs, optimizer):
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,       
                 num_warmup_steps=0, num_training_steps=total_steps)
    return scheduler

def median_absolute_percentage_error(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true))

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
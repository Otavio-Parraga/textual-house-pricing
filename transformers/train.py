import torch
from evaluation import evaluate
from utils import get_scheduler
from transformers import logging

logging.set_verbosity_error()


def train(model, train_loader, test_loader, optimizer, criterion, device, epochs):

    scheduler = get_scheduler(train_loader, epochs, optimizer)

    for epoch in range(epochs):
        total_rmse = 0
        total_loss = 0

        model.train()

        for i, (text, price) in enumerate(train_loader):
            model.zero_grad()

            input_ids = text['input_ids'].to(device)
            attention_mask = text['attention_mask'].to(device)
            price = price.to(device)

            output = model(input_ids, attention_mask)

            output = output.float().squeeze()
            price = price.float().squeeze()

            loss = criterion(output, price)

            total_loss += loss.item()
            total_rmse += torch.sum(torch.sqrt(((torch.exp(price) - torch.exp(output))**2)))

            print(
                f'\rEpoch {epoch+1} | Iter: {i+1}/{len(train_loader)} | RMSE: {total_rmse / (i+1) * train_loader.batch_size} | Loss: {total_loss / (i+1)}', end='')


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            scheduler.step()

        print(evaluate(model, test_loader, device))
            # TODO: save best model
        #save_model(model, f'checkpoint-epoch{epoch+1}')

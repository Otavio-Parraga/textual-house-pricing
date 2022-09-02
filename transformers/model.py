import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def load_models(pretrained_name, **kwargs):
    return AutoModel.from_pretrained(pretrained_name, **kwargs), AutoTokenizer.from_pretrained(pretrained_name, **kwargs)


def save_model(model, name):
    torch.save(model.state_dict(), f'./trained_models/{name}.pt')


def load_checkpoint(pretrained_name, path):
    model = RegressorTransformer(
        pretrained_name, cache_dir=f'./pretrained_save/{pretrained_name}')
    model.load_state_dict(torch.load(path))
    return model


class RegressorTransformer(nn.Module):
    def __init__(self, pretrained_model, dropout=0.2, **kwargs):
        super(RegressorTransformer, self).__init__()
        self.encoder, self.tokenizer = load_models(
            pretrained_name=pretrained_model, **kwargs)
        self.regressor = nn.Sequential(nn.Dropout(dropout),
                                       nn.Linear(768, 1))

    def forward(self, input_ids, attention_masks):
        encoded_repr = self.encoder(input_ids, attention_masks)
        output = self.regressor(encoded_repr[1])
        return output

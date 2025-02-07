# Code adapted from https://github.com/huiwenn/CopulaCPTS

import torch
import torch.nn as nn

from tqdm import trange


class CopulaCPTSModule(nn.Module):
    def __init__(self, dimension, alpha):
        super().__init__()
        self.alpha = alpha
        self.alphas = nn.Parameter(torch.ones(dimension))
        self.relu = torch.nn.ReLU()

    def forward(self, pseudo_data):
        coverage = torch.mean(
            torch.relu(
                torch.prod(
                    torch.sigmoid((self.alphas - pseudo_data) * 1000)
                , dim=1)
            )
        )
        return torch.abs(coverage - (1 - self.alpha))


def search_alpha(pseudo_data, alpha, epochs):
    dim = pseudo_data.shape[-1]
    module = CopulaCPTSModule(dim, alpha)
    optimizer = torch.optim.AdamW(module.parameters(), weight_decay=1e-4)

    with trange(epochs, desc='training', unit='epochs', bar_format='{desc}: {n}{postfix}') as pbar:
        for i in pbar:
            optimizer.zero_grad()
            loss = module(pseudo_data)

            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                pbar.set_postfix(loss=loss.detach().numpy())
            if loss < 5e-6:
                break

    return module.alphas.detach()

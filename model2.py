import torch.nn as nn
import torch
import torch.nn.functional as F

from math import floor, ceil


class NNModel(nn.Module):
    sequential_model = None

    def __init__(self, gen):
        super().__init__()
        self.gen = gen
        self.init_gen()

    def init_gen(self):
        seq_layers = []

        for layer in self.gen["layers"][1:]:
            l_type = layer["type"]

            if l_type == "Conv2d":
                seq_layers.append(nn.Conv2d(layer["in_channels"], layer["out_channels"], layer["kernel_size"], padding=layer["padding"]))

            if l_type == "Linear":
                seq_layers.append(nn.Linear(layer["in"], layer["out"]))

            if l_type == "MaxPool2d":
                seq_layers.append(nn.MaxPool2d(layer["size"], layer["size"]))

            if l_type == "ReLU":
                seq_layers.append(nn.ReLU())

            if l_type == "Flatten":
                seq_layers.append(nn.Flatten())

        self.sequential_model = nn.Sequential(*self.seq_layers)

    def forward(self, x):
        scores = self.layers(x)
        return scores


class Environment:
    entities = None

    train_loader = None
    validation_loader = None
    test_loader = None

    epochs = None
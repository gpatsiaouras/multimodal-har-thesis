import torch
import torch.nn as nn


class ELM:
    def __init__(self, input_size, hidden_size, num_classes, device=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.device = device

        self.alpha = nn.init.uniform_(torch.empty(self.input_size, self.hidden_size, device=self.device), a=-1., b=1.)
        self.beta = nn.init.uniform_(torch.empty(self.hidden_size, self.num_classes, device=self.device), a=-1., b=1.)

        self.bias = torch.zeros(self.hidden_size, device=self.device)

        self.activation = nn.ReLU()

    def predict(self, x):
        h = self.activation(torch.add(x.mm(self.alpha), self.bias))
        out = h.mm(self.beta)

        return out

    def fit(self, x, t):
        temp = x.mm(self.alpha)
        h = self.activation(torch.add(temp, self.bias))

        h_pinv = torch.pinverse(h)
        self.beta = h_pinv.mm(t)

    def evaluate(self, x, t):
        y_pred = self.predict(x)
        acc = torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(t, dim=1)).item() / len(t)
        return acc

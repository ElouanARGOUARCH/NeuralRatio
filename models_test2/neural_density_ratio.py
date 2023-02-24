import torch
from torch import nn
from tqdm import tqdm

class NDR(nn.Module):
    def __init__(self, target_samples_1, target_samples_2, hidden_dims):
        super().__init__()

        self.target_samples_2 = target_samples_1
        self.target_samples_2 = target_samples_2
        self.p = target_samples_1.shape[-1]

        network_dimensions = [self.p] + hidden_dims + [1]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1), nn.SiLU(), ])
        network.pop()
        self.logit_r = nn.Sequential(*network)

        self.loss_values=[]

    def loss(self, X,Y):
        log_sigmoid = torch.nn.LogSigmoid()
        return -torch.mean(log_sigmoid(self.logit_r(X))+log_sigmoid(-self.logit_r(Y)))

    def train(self, epochs):
        self.para_list = list(self.parameters())

        self.optimizer = torch.optim.Adam(self.para_list, lr=5e-3)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        X = self.target_samples_1.to(device)
        Y = self.target_samples_2.to(device)

        pbar = tqdm(range(epochs))
        for t in pbar:
            self.optimizer.zero_grad()
            batch_loss = self.loss(X,Y)
            batch_loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor(self.loss(X,Y)).item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)))
        self.to(torch.device('cpu'))
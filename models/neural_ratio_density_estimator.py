import torch
from torch import nn
from tqdm import tqdm

class NDRE(nn.Module):
    def __init__(self, target_samples, hidden_dims):
        super().__init__()

        self.target_samples = target_samples
        self.p = target_samples.shape[-1]

        network_dimensions = [self.p] + hidden_dims + [1]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1), nn.SiLU(), ])
        network.pop()
        self.logit_r = nn.Sequential(*network)

        if self.p >= 2:
            cov = torch.cov(self.target_samples.T)
        else:
            cov = torch.var(self.target_samples, dim=0) * torch.eye(self.p)
        self.reference = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.mean(self.target_samples, dim=0), cov)

        self.loss_values=[]

    def log_density(self, x):
        return self.logit_r(x).squeeze(-1) + self.reference.log_prob(x)

    def loss(self, X):
        log_sigmoid = torch.nn.LogSigmoid()
        true = X
        fake = self.reference.sample(X.shape[:-1])
        return -torch.mean(log_sigmoid(self.logit_r(true))+log_sigmoid(-self.logit_r(fake)))

    def train(self, epochs, batch_size = None):
        self.para_list = list(self.parameters())

        self.optimizer = torch.optim.Adam(self.para_list, lr=5e-3)
        if batch_size is None:
            batch_size = self.target_samples.shape[0]
        dataset = torch.utils.data.TensorDataset(self.target_samples)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        pbar = tqdm(range(epochs))
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                x = batch[0].to(device)
                self.optimizer.zero_grad()
                batch_loss = self.loss(x)
                batch_loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor([self.loss(batch[0].to(device)) for i, batch in enumerate(dataloader)]).mean().item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)))
        self.to(torch.device('cpu'))
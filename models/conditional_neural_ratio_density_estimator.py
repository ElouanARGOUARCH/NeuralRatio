import torch
from torch import nn
from tqdm import tqdm

class CNDRE(nn.Module):
    def __init__(self, x_samples, theta_samples, hidden_dims, mode = 'Proxy'):
        super().__init__()

        self.p = x_samples.shape[-1]
        self.d = theta_samples.shape[-1]
        self.x_samples = x_samples
        self.num_samples = x_samples.shape[0]
        assert self.num_samples == theta_samples.shape[0], "Number of samples do not match"
        self.theta_samples = theta_samples

        network_dimensions = [self.p + self.d] + hidden_dims + [1]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1), nn.Sigmoid(), ])
        network.pop()
        self.logit_r = nn.Sequential(*network)

        self.mode = mode
        if self.mode == 'Proxy':
            if self.p >= 2:
                cov = torch.cov(self.x_samples.T)
            else:
                cov = torch.var(self.x_samples, dim=0) * torch.eye(self.p)
            self.reference = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.mean(self.x_samples, dim=0), cov)

        self.loss_values = []

    def loss(self, x, theta):
        log_sigmoid = torch.nn.LogSigmoid()
        if self.mode == 'Proxy':
            x_tilde = self.reference.sample([x.shape[0]])
        elif self.mode =='Ratio':
            x_tilde = x[torch.randperm(x.shape[0])]
        true = torch.cat([x, theta], dim=-1)
        fake = torch.cat([x_tilde, theta], dim=-1)
        return -torch.mean(log_sigmoid(self.logit_r(true)) + log_sigmoid(-self.logit_r(fake)))

    def log_density(self,x, t):
        assert self.mode == 'Proxy', "log_density requires Proxy mode"
        logit_r = self.logit_r(torch.cat([x, t], dim=-1)).squeeze(-1)
        return logit_r + self.reference.log_prob(x)

    def log_ratio(self,x,t):
        assert self.mode == 'Ratio', "log_ratio requires Ratio mode"
        return self.logit_r(torch.cat([x, t], dim=-1)).squeeze(-1)

    def train(self, epochs, batch_size):
        self.para_list = list(self.parameters())

        self.optimizer = torch.optim.Adam(self.para_list, lr=5e-3)
        if batch_size is None:
            batch_size = self.target_samples.shape[0]
        dataset = torch.utils.data.TensorDataset(self.theta_samples, self.x_samples)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        pbar = tqdm(range(epochs))
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                theta = batch[0].to(device)
                x = batch[1].to(device)
                self.optimizer.zero_grad()
                batch_loss = self.loss(x, theta)
                batch_loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor([self.loss(batch[1].to(device),batch[0].to(device)) for i, batch in enumerate(dataloader)]).mean().item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)))
        self.to(torch.device('cpu'))


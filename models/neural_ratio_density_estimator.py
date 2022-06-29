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
        self.log_constant = None

    def log_density(self, x):
        return self.log_density_ratio(x) + self.reference.log_prob(x)

    def log_density_ratio(self,x):
        return self.logit_r(x).squeeze(-1)

    def estimate_constant(self):
        cat = torch.cat([self.target_samples, self.reference.sample([self.target_samples.shape[0]])], dim =0)
        self.log_constant = torch.max(self.log_density_ratio(cat))

    def sample(self, num_samples):
        if self.log_constant is None:
            print('estimating rejection sampling constant')
            self.estimate_constant()
        fake = self.reference.sample([num_samples])
        log_density_ratios = self.log_density_ratio(fake)
        accepted = torch.log(torch.rand(num_samples)) < log_density_ratios - self.log_constant
        return fake[accepted]

    def loss(self, X):
        log_sigmoid = torch.nn.LogSigmoid()
        return -torch.mean(log_sigmoid(self.logit_r(X))+log_sigmoid(-self.logit_r(self.reference.sample(X.shape[:-1]))))

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
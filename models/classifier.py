import torch
from torch.distributions import Categorical
from tqdm import tqdm
from torch import nn

class EMDensityEstimator(nn.Module):
    def __init__(self,target_samples,K):
        super().__init__()
        self.target_samples = target_samples
        self.p = self.target_samples.shape[-1]
        self.K = K
        self.log_pi = torch.log(torch.ones([self.K])/self.K)
        self.m = self.target_samples[torch.randint(low= 0, high = self.target_samples.shape[0],size = [self.K])]
        self.log_s = torch.log(torch.var(self.target_samples, dim = 0)).unsqueeze(0).repeat(self.K, 1)/2

        self.loss_values = []

    def forward(self, x):
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        X = x.unsqueeze(-2).expand(desired_size)
        return (X - self.m.expand_as(X)) / torch.exp(self.log_s).expand_as(X)

    def backward(self,z):
        desired_size = list(z.shape)
        desired_size.insert(-1, self.K)
        Z = z.unsqueeze(-2).expand(desired_size)
        return Z * torch.exp(self.log_s).expand_as(Z) + self.m.expand_as(Z)

    def log_det_J(self,x):
        return -torch.sum(self.log_s, dim = -1)

    def compute_log_v(self,x):
        z = self.forward(x)
        unormalized_log_v = self.reference_log_density(z) + self.log_pi.unsqueeze(0).repeat(x.shape[0],1)+ self.log_det_J(x)
        return unormalized_log_v - torch.logsumexp(unormalized_log_v, dim = -1, keepdim= True)

    def sample_latent(self,x):
        z = self.forward(x)
        pick = Categorical(torch.exp(self.compute_log_v(x))).sample()
        return torch.stack([z[i,pick[i],:] for i in range(x.shape[0])])

    def sample_reference(self, num_samples):
        return torch.distributions.MultivariateNormal(torch.zeros(self.p), torch.eye(self.p)).sample(num_samples)

    def reference_log_density(self, z):
        return -torch.sum(torch.square(z)/2, dim = -1) - torch.log(torch.tensor([2*torch.pi], device = z.device))*self.p/2

    def log_density(self, x):
        z = self.forward(x)
        return torch.logsumexp(self.reference_log_density(z) + self.log_pi.unsqueeze(0).repeat(x.shape[0],1) + self.log_det_J(x),dim=-1)

    def sample(self, num_samples):
        z = self.sample_reference(num_samples)
        x = self.backward(z)
        pick = Categorical(torch.exp(self.log_pi.unsqueeze(0).repeat(x.shape[0],1))).sample()
        return torch.stack([x[i,pick[i],:] for i in range(z.shape[0])])

    def M_step(self, batch):
        v = torch.exp(self.compute_log_v(batch))
        c = torch.sum(v, dim=0)
        self.log_pi = torch.log(c) - torch.logsumexp(torch.log(c), dim = 0)
        self.m = torch.sum(v.unsqueeze(-1).repeat(1, 1, self.p) * batch.unsqueeze(-2).repeat(1, self.K, 1),
                                dim=0) / c.unsqueeze(-1)
        temp = batch.unsqueeze(1).repeat(1,self.K, 1) - self.m.unsqueeze(0).repeat(batch.shape[0],1,1)
        temp2 = torch.square(temp)
        self.log_s = torch.log(torch.sum(v.unsqueeze(-1).repeat(1, 1, self.p) * temp2,dim=0)/c.unsqueeze(-1))/2

    def train(self, epochs):
        pbar = tqdm(range(epochs))
        for t in pbar:
            self.M_step(self.target_samples)
            iteration_loss = -torch.mean(self.log_density(self.target_samples)).detach().item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(iteration_loss))
    def sample_joint(self, num_samples):
        z = self.sample_reference(num_samples)
        x = self.backward(z)
        pick = Categorical(torch.exp(self.log_pi.unsqueeze(0).repeat(x.shape[0],1))).sample()
        return torch.stack([x[i,pick[i],:] for i in range(z.shape[0])]), pick

class SoftmaxWeight(nn.Module):
    def __init__(self, K, p, hidden_dimensions =[]):
        super().__init__()
        self.K = K
        self.p = p
        self.network_dimensions = [self.p] + hidden_dimensions + [self.K]
        network = []
        for h0, h1 in zip(self.network_dimensions, self.network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1),nn.Tanh(),])
        network.pop()
        self.f = nn.Sequential(*network)

    def log_prob(self, z):
        unormalized_log_w = self.f.forward(z)
        return unormalized_log_w - torch.logsumexp(unormalized_log_w, dim=-1, keepdim=True)

class classifier(nn.Module):
    def __init__(self, K,samples, labels, hidden_dims= []):
        super().__init__()
        self.K = K
        self.samples = samples
        self.labels= labels
        self.model = SoftmaxWeight(K,self.samples.shape[-1], hidden_dims)

    def loss(self, samples,labels):
        return -torch.sum((self.model.log_prob(samples))[range(samples.shape[0]), labels])

    def train(self, epochs, lr = 5e-3):
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pbar = tqdm(range(epochs))
        for _ in pbar:
            optim.zero_grad()
            loss = self.loss(self.samples.to(device), self.labels.to(device))
            loss.backward()
            optim.step()
            pbar.set_postfix_str('loss = ' + str(round(loss.item(),4)) + '; device = ' + str(device))



import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.color_visual import *
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

class GenerativeAdversarialNetwork(nn.Module):
    def __init__(self, target_samples, latent_dim, networks_dims):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p = target_samples.shape[-1]
        self.target_samples = target_samples
        self.num_samples = target_samples.shape[0]

        network_logit_r_dimensions = [self.p] + networks_dims + [1]
        network_logit_r = []
        for h0, h1 in zip(network_logit_r_dimensions, network_logit_r_dimensions[1:]):
            network_logit_r.extend([nn.Linear(h0, h1), nn.SiLU(), ])
        network_logit_r.pop()
        self.logit_r_network = nn.Sequential(*network_logit_r).to(self.device)
        self.optimizer_logit_r = torch.optim.Adam(self.logit_r_network.parameters(), lr=5e-3)

        self.d = latent_dim
        network_generator_dimensions = [self.d] + networks_dims + [self.p]
        network_generator = []
        for h0, h1 in zip(network_generator_dimensions, network_generator_dimensions[1:]):
            network_generator.extend([nn.Linear(h0, h1), nn.SiLU(), ])
        self.generator_network = nn.Sequential(*network_generator).to(self.device)
        self.optimizer_generator = torch.optim.Adam(self.generator_network.parameters(), lr= 5e-3)

        self.to(self.device)
        #self.log_sigmoid = torch.nn.LogSigmoid()

    def sample_generator(self, num_samples):
        z = torch.randn(num_samples,self.d).to(self.device)
        return self.generator_network(z)

    def loss_discriminator(self, X):
        log_sigmoid = torch.nn.LogSigmoid()
        true = X
        fake = self.sample_generator(X.shape[0])
        return -torch.mean(log_sigmoid(self.logit_r_network(true))+log_sigmoid(-self.logit_r_network(fake)))

    def loss_generator(self, X):
        log_sigmoid = torch.nn.LogSigmoid()
        fake = self.sample_generator(X.shape[0])
        return -torch.mean(log_sigmoid(-self.logit_r_network(fake)))

    def train(self, epochs, batch_size = None):
        if batch_size is None:
            batch_size = self.num_samples
        loss_values = [torch.mean(self.loss_discriminator(self.target_samples)).item()]
        pbar = tqdm(range(epochs))
        for t in pbar:
            perm = torch.randperm(self.num_samples)
            for i in range(int(self.num_samples / batch_size) + 1 * (int(self.num_samples / batch_size) != self.num_samples / batch_size)):
                self.optimizer_logit_r.zero_grad()
                batch_loss_logit_r= self.loss_discriminator(self.target_samples[perm][i * batch_size:min((i + 1) * batch_size, self.num_samples)])
                batch_loss_logit_r.backward()
                self.optimizer_logit_r.step()

            for i in range(int(self.num_samples / batch_size) + 1 * (int(self.num_samples / batch_size) != self.num_samples / batch_size)):
                self.optimizer_generator.zero_grad()
                batch_loss_generator = - self.loss_generator(self.target_samples[perm][i * batch_size:min((i + 1) * batch_size, self.num_samples)])
                batch_loss_generator.backward()
                self.optimizer_generator.step()

            iteration_loss = torch.mean(self.loss_discriminator(self.target_samples)).item()
            loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(iteration_loss))

    def model_visual(self, num_samples = 5000):
        if self.p == 1:
            plt.figure()
            plt.hist(self.target_samples[:num_samples].cpu().numpy(), color = 'red', bins = 200)
            plt.figure()
            plt.hist(self.sample_generator(num_samples).cpu().detach().numpy(),  color = 'blue', bins = 200)


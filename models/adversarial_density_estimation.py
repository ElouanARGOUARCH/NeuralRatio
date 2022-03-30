import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.color_visual import *
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

class RealNVP(nn.Module):
    def __init__(self,p,hidden_dim):

        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p = p
        net = []
        hs = [self.p] + hidden_dim + [2*self.p]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
                nn.Tanh(),
            ])
        net.pop()
        self.net = nn.Sequential(*net)

        self.mask = [torch.cat([torch.zeros(int(self.p/2)), torch.ones(self.p - int(self.p/2))], dim = 0).to(self.device),torch.cat([torch.ones(int(self.p/2)), torch.zeros(self.p - int(self.p/2))], dim = 0).to(self.device)]
        self.lr = 5e-5
        self.to(self.device)

    def estimate_moments(self, target_samples):
        if self.p >= 2:
            cov = torch.cov(target_samples.T)
        else:
            cov = torch.var(target_samples, dim=0) * torch.eye(self.p)
        self.q_distribution= torch.distributions.multivariate_normal.MultivariateNormal(
            torch.mean(target_samples, dim=0), cov)


    def sample_forward(self,x):
        z = x
        for mask in reversed(self.mask):
            out = self.net(mask * z)
            m, log_s = out[...,:self.p]*(1 - mask),out[...,self.p:]*(1 - mask)
            z = (z*(1 - mask) * torch.exp(log_s)+m) + (mask * z)
        return z

    def sample(self, num_samples):
        return self.sample_backward(self.q_distribution.sample(num_samples))

    def sample_backward(self, z):
        x = z
        for mask in self.mask:
            out = self.net(x*mask)
            m, log_s = out[...,:self.p]* (1 - mask),out[...,self.p:]* (1 - mask)
            x = ((x*(1-mask) -m)/torch.exp(log_s)) + (x*mask)
        return x

    def log_prob(self, x):
        z = x
        log_det = torch.zeros(x.shape[:-1]).to(self.device)
        for mask in reversed(self.mask):
            out = self.net(mask * z)
            m, log_s = out[...,:self.p]* (1 - mask),out[...,self.p:]* (1 - mask)
            z = (z*(1 - mask)*torch.exp(log_s) + m) + (mask*z)
            log_det += torch.sum(log_s, dim = -1)
        return self.q_distribution.log_prob(z) + log_det

class AdversarialDensityEstimator(nn.Module):
    def __init__(self, target_samples, hidden_dims):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p = target_samples.shape[-1]
        self.target_samples = target_samples
        self.num_samples = target_samples.shape[0]
        network_dimensions = [self.p] + hidden_dims + [1]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1), nn.SiLU(), ])
        network.pop()
        self.logit_r = nn.Sequential(*network).to(self.device)
        self.optimizer_discrimator = torch.optim.Adam(self.logit_r.parameters(), lr=5e-3)
        self.reference = RealNVP(self.p, [256,256,256]).to(self.device)
        self.reference.estimate_moments(self.target_samples)
        self.optimizer_reference = torch.optim.Adam(self.reference.parameters(), lr= 5e-3)
        self.to(self.device)

    def log_density(self, x):
        return self.logit_r(x).squeeze(-1) + self.reference.log_prob(x)

    def loss(self, X):
        log_sigmoid = torch.nn.LogSigmoid()
        true = X
        fake = self.reference.sample(X.shape[:-1])
        return -torch.mean(log_sigmoid(self.logit_r(true))+log_sigmoid(-self.logit_r(fake)))

    def train(self, epochs, batch_size = None):
        if batch_size is None:
            batch_size = self.num_samples
        loss_values = [torch.mean(self.loss(self.target_samples)).item()]
        best_loss = loss_values[0]
        best_iteration = 0
        pbar = tqdm(range(epochs))
        for t in pbar:
            perm = torch.randperm(self.num_samples)
            for i in range(
                    int(self.num_samples / batch_size) + 1 * (int(self.num_samples / batch_size) != self.num_samples / batch_size)):
                self.optimizer_discrimator.zero_grad()
                batch_loss_discriminator = self.loss(self.target_samples[perm][i * batch_size:min((i + 1) * batch_size, self.num_samples)])
                batch_loss_discriminator.backward()
                self.optimizer_discrimator.step()

                self.optimizer_reference.zero_grad()
                batch_loss_reference = - self.loss(self.target_samples[perm][i * batch_size:min((i + 1) * batch_size, self.num_samples)])
                batch_loss_reference.backward()
                self.optimizer_reference.step()

            iteration_loss = torch.mean(self.loss(self.target_samples)).item()
            loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(iteration_loss))
            if iteration_loss < best_loss:
                best_loss = iteration_loss
                best_iteration = t + 1
                #best_parameters = self.state_dict()
        #self.load_state_dict(best_parameters)
        self.train_visual(best_loss, best_iteration, loss_values)

    def train_visual(self, best_loss, best_iteration, loss_values):
        fig = plt.figure(figsize=(12, 4))
        ax = plt.subplot(111)
        Y1, Y2 = best_loss - (max(loss_values) - best_loss) / 2, max(loss_values) + (max(loss_values) - best_loss) / 4
        ax.set_ylim(Y1, Y2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(loss_values, label='Loss values during training', color='black')
        ax.scatter([best_iteration], [best_loss], color='black', marker='d')
        ax.axvline(x=best_iteration, ymax=(best_loss - best_loss + (max(loss_values) - best_loss) / 2) / (
                    max(loss_values) + (max(loss_values) - best_loss) / 4 - best_loss + (
                        max(loss_values) - best_loss) / 2), color='black', linestyle='--')
        ax.text(0, best_loss - (max(loss_values) - best_loss) / 8,
                'best iteration = ' + str(best_iteration) + '\nbest loss = ' + str(best_loss),
                verticalalignment='top', horizontalalignment='left', fontsize=12)
        if len(loss_values) > 30:
            x1, x2 = best_iteration - int(len(loss_values) / 15), min(best_iteration + int(len(loss_values) / 15),
                                                                      len(loss_values) - 1)
            k = len(loss_values) / (2.5 * (x2 - x1 + 1))
            offset = (Y2-Y1)/(6*k)
            y1, y2 = best_loss - offset, best_loss + offset
            axins = zoomed_inset_axes(ax, k, loc='upper right')
            axins.axvline(x=best_iteration, ymax=(best_loss - y1) / (y2-y1), color='black', linestyle='--')
            axins.scatter([best_iteration], [best_loss], color='black', marker='d')
            axins.xaxis.set_major_locator(MaxNLocator(integer=True))
            axins.plot(loss_values, color='black')
            axins.set_xlim(x1 - .5, x2 + .5)
            axins.set_ylim(y1, y2)
            mark_inset(ax, axins, loc1=3, loc2=4)

    def model_visual(self):
        if self.p == 1:
            tt = torch.linspace(torch.min(self.target_samples), torch.max(self.target_samples), 400).unsqueeze(-1)
            fig = plt.figure(figsize=(12, 8))
            plt.hist(self.target_samples.squeeze(-1).numpy(), density=True, bins=150, color='red', label='target samples',
                     alpha=.6)
            plt.plot(tt, torch.exp(self.log_density(tt)).detach().numpy(), color='blue', label='model density')
            plt.plot(tt, torch.exp(self.reference.log_prob(tt)).detach().numpy(), color='green',
                     label='reference density')
            plt.hist(self.reference.sample([50000]).cpu().detach().squeeze(-1).numpy(), density=True, bins=150, color='green',
                     label='reference samples', alpha=.6)
            fig.legend()

        elif self.p == 2:
            plt.figure()
            proxy_samples = self.reference.sample([self.num_samples])
            plt.scatter(proxy_samples[:, 0].cpu().detach().numpy(), proxy_samples[:, 1].cpu().detach().numpy(), color='green', alpha = .4)
            plt.figure()
            print(torch.min(proxy_samples[:, 0]))
            tt_0 = torch.linspace(torch.min(proxy_samples[:, 0]).detach().item(), torch.max(proxy_samples[:, 0]).detach().item(), 500)
            tt_1 = torch.linspace(torch.min(proxy_samples[:, 1]).detach().item(), torch.max(proxy_samples[:, 1]).detach().item(), 500)
            grid = torch.cartesian_prod(tt_1, tt_0)
            density = torch.exp(self.reference.log_prob(grid)).reshape(500, 500).T
            plt.pcolormesh(tt_1, tt_0, density.detach().numpy(), cmap=green_cmap)
            plt.figure()
            plt.scatter(self.target_samples[:, 0], self.target_samples[:, 1], color='red', alpha = .4)
            plt.figure()
            tt_0 = torch.linspace(torch.min(self.target_samples[:, 0]), torch.max(self.target_samples[:, 0]), 500)
            tt_1 = torch.linspace(torch.min(self.target_samples[:, 1]), torch.max(self.target_samples[:, 1]), 500)
            grid = torch.cartesian_prod(tt_1, tt_0)
            density = torch.exp(self.log_density(grid)).reshape(500, 500).T
            plt.pcolormesh(tt_1, tt_0, density.detach().numpy(), cmap=blue_cmap)

import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.color_visual import *
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


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
        self.optimizer = torch.optim.Adam(self.logit_r.parameters(), lr=1e-3)
        self.mode = mode
        if self.mode == 'Proxy':
            if self.p >= 2:
                cov = torch.cov(self.x_samples.T)
            else:
                cov = torch.var(self.x_samples, dim=0) * torch.eye(self.p)
            self.reference = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.mean(self.x_samples, dim=0), cov)

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
        logit_r = self.logit_r(torch.cat([x, t], dim=-1)).squeeze(-1)
        if self.mode =='Proxy':
            return logit_r + self.reference.log_prob(x)
        elif self.mode == 'Ratio':
            return logit_r

    def model_visual(self):
        if self.p==1 and self.d==1:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot()
            ax.set_xlabel('theta')
            ax.set_ylabel('x')
            delta = 500
            x_tt = torch.linspace(torch.min(self.x_samples), torch.max(self.theta_samples), delta)
            theta_tt = torch.linspace(torch.min(self.theta_samples), torch.max(self.theta_samples), delta)
            ax.pcolormesh(x_tt,theta_tt,torch.exp(self.log_density(x_tt.unsqueeze(-1).unsqueeze(1).repeat(1, delta, 1), theta_tt.unsqueeze(-1).unsqueeze(0).repeat(delta, 1, 1))).detach().numpy(), cmap = blue_cmap)

    def train(self, epochs, batch_size):
        loss_values = [self.loss(self.x_samples, self.theta_samples).item()]
        best_loss = loss_values[0]
        best_iteration = 0
        best_parameters = self.state_dict()
        pbar = tqdm(range(epochs))
        for t in pbar:
            perm = torch.randperm(self.num_samples)
            for i in range(int(self.num_samples / batch_size) + 1 * (
                    int(self.num_samples / batch_size) != self.num_samples / batch_size)):
                self.optimizer.zero_grad()
                batch_loss = self.loss(self.x_samples[perm][i * batch_size:min((i + 1) * batch_size, self.num_samples)],
                                       self.theta_samples[perm][i * batch_size:min((i + 1) * batch_size, self.num_samples)])
                batch_loss.backward()
                self.optimizer.step()
            iteration_loss = self.loss(self.x_samples, self.theta_samples).item()
            loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(iteration_loss))
            if iteration_loss < best_loss:
                best_loss = iteration_loss
                best_iteration = t + 1
                best_parameters = self.state_dict()
        self.load_state_dict(best_parameters)
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
                'best iteration = ' + str(best_iteration) + '\nbest loss = ' + str(np.round(best_loss, 5)),
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


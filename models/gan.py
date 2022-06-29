import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

class GAN(nn.Module):
    def __init__(self, target_samples,d,hidden_dims):
        super().__init__()

        self.target_samples = target_samples
        self.p = target_samples.shape[-1]
        self.d = d

        discriminator_network_dimensions = [self.p] + hidden_dims + [1]
        discriminator_network = []
        for h0, h1 in zip(discriminator_network_dimensions, discriminator_network_dimensions[1:]):
            discriminator_network.extend([nn.Linear(h0, h1), nn.SiLU(), ])
        discriminator_network.pop()
        self.logit_r = nn.Sequential(*discriminator_network)

        generator_network_dimensions = [self.d] + hidden_dims + [self.p]
        generator_network = []
        for h0, h1 in zip(generator_network_dimensions, generator_network_dimensions[1:]):
            generator_network.extend([nn.Linear(h0, h1), nn.Sigmoid(), ])
        generator_network.pop()
        self.generator = nn.Sequential(*generator_network)

        self.loss_values=[]
        self.log_constant = None

    def sample_generator(self, num_samples):
        noise = torch.randn([num_samples, self.d])
        return self.generator(noise)

    def log_density_ratio(self,x):
        return self.logit_r(x).squeeze(-1)

    def estimate_constant(self):
        cat = torch.cat([self.target_samples, self.sample_generator(self.target_samples.shape[0])], dim =0)
        self.log_constant = torch.max(self.log_density_ratio(cat))

    def sample(self, num_samples):
        if self.log_constant is None:
            print('estimating rejection sampling constant')
            self.estimate_constant()
        fake = self.sample_generator(num_samples)
        log_density_ratios = self.log_density_ratio(fake)
        accepted = torch.log(torch.rand(num_samples)) < log_density_ratios - self.log_constant
        return fake[accepted]

    def loss_discriminator(self, X):
        log_sigmoid = torch.nn.LogSigmoid()
        return -torch.mean(log_sigmoid(self.logit_r(X))+log_sigmoid(-self.logit_r(self.sample_generator(X.shape[0]))))

    def loss_generator(self, X):
        log_sigmoid = torch.nn.LogSigmoid()
        return -torch.mean(self.logit_r(self.sample_generator(X.shape[0])))

    def train(self, epochs, batch_size = None):
        self.generator_para_list = list(self.generator.parameters())
        self.generator_optimizer = torch.optim.SGD(self.generator_para_list, lr=5e-3)

        self.discriminator_para_list = list(self.logit_r.parameters())
        self.discriminator_optimizer = torch.optim.SGD(self.discriminator_para_list, lr=5e-3)

        if batch_size is None:
            batch_size = self.target_samples.shape[0]
        pbar = tqdm(range(epochs))
        for t in pbar:
            for micro_epochs in range(50):
                self.discriminator_optimizer.zero_grad()
                discriminator_batch_loss = self.loss_discriminator(self.target_samples)
                discriminator_batch_loss.backward()
                self.discriminator_optimizer.step()
            fig = plt.figure()
            ax = fig.add_subplot(121)
            ax.scatter(self.target_samples[:,0].numpy(), self.target_samples[:,1].numpy(), color = 'red')
            with torch.no_grad():
                generator_samples = self.sample_generator(self.target_samples.shape[0])
            ax = fig.add_subplot(122)
            ax.scatter(generator_samples[:, 0].numpy(), generator_samples[:, 1].numpy(), color='blue')
            plt.show()
            for micro_epochs in range(50):
                self.generator_optimizer.zero_grad()
                generator_batch_loss = self.loss_generator(self.target_samples)
                generator_batch_loss.backward()
                self.generator_optimizer.step()
            with torch.no_grad():
                iteration_loss = self.loss_generator(self.target_samples).item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)))
        self.to(torch.device('cpu'))
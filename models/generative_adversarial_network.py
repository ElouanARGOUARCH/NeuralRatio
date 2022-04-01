import torch
from torch import nn
from tqdm import tqdm

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
        network_logit_r.pop()
        self.generator_network = nn.Sequential(*network_generator).to(self.device)
        self.optimizer_generator = torch.optim.Adam(self.generator_network.parameters(), lr= 5e-3)

        self.log_sigmoid = torch.nn.LogSigmoid()
        self.to(self.device)

    def sample_generator(self, num_samples):
        z = torch.randn(num_samples,self.d).to(self.device)
        return self.generator_network(z)

    def loss_discriminator(self, X):
        true = X
        fake = self.sample_generator(X.shape[0])
        return -torch.mean(self.log_sigmoid(self.logit_r_network(true))+self.log_sigmoid(-self.logit_r_network(fake)))

    def loss_generator(self, X):
        fake = self.sample_generator(X.shape[0])
        return -torch.mean(self.log_sigmoid(-self.logit_r_network(fake)))

    def train(self, epochs, batch_size = None):
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
                self.optimizer_logit_r.zero_grad()
                batch_loss_logit_r= self.loss_discriminator(x)
                batch_loss_logit_r.backward()
                self.optimizer_logit_r.step()
                self.optimizer_generator.zero_grad()
                batch_loss_generator = - self.loss_generator(x)
                batch_loss_generator.backward()
                self.optimizer_generator.step()
            with torch.no_grad():
                iteration_logit_r_loss = torch.tensor([self.loss_discriminator(batch[0].to(device)) for i, batch in enumerate(dataloader)]).mean().item()
                iteration_generator_loss = torch.tensor([self.loss_generator(batch[0].to(device)) for i, batch in enumerate(dataloader)]).mean().item()
            pbar.set_postfix_str('iteration_logit_r_loss = ' + str(iteration_logit_r_loss) + str(" ; iteration_generator_loss = ") + str(iteration_generator_loss))
        self.to(torch.device('cpu'))

    def model_visual(self, num_samples = 5000):
        if self.p == 1:
            plt.figure()
            plt.hist(self.target_samples[:num_samples].numpy(), color = 'red', bins = 200)
            with torch.no_grad():
                plt.figure()
                plt.hist(self.sample_generator(num_samples).numpy(),  color = 'blue', bins = 200)
        if self.p == 2:
            plt.figure()
            plt.scatter(self.target_samples[:num_samples,0].numpy(),self.target_samples[:num_samples,1].numpy(), color = 'red', alpha = 0.4, label = 'target_samples')
            with torch.no_grad():
                samples = self.sample_generator(num_samples)
                plt.figure()
                plt.scatter(samples[:num_samples, 0].numpy(), samples[:num_samples,1].numpy(),color = 'blue', alpha = 0.4, label = 'model_samples')

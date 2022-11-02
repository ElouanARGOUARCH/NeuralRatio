import torch
from torch import nn
from tqdm import tqdm

class EvenBinaryClassifier(nn.Module):
    def __init__(self, label_1_samples, label_0_samples, hidden_dims):
        super().__init__()

        self.label_1_samples = label_1_samples
        self.label_0_samples = label_0_samples
        assert label_0_samples.shape[0]==label_1_samples.shape[0], 'mismatch in number of samples'
        self.N = label_1_samples.shape[0]
        assert label_0_samples.shape[-1]==label_1_samples.shape[-1], 'mismatch in samples dimensions'
        self.p = label_0_samples.shape[-1]

        network_dimensions = [self.p] + hidden_dims + [1]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1), nn.SiLU(), ])
        network.pop()
        self.logit_r = nn.Sequential(*network)

        self.loss_values=[]


    def loss(self, label_1_batch, label_0_batch):
        log_sigmoid = torch.nn.LogSigmoid()
        return -torch.mean(log_sigmoid(self.logit_r(label_1_batch)) + log_sigmoid(-self.logit_r(label_0_batch)))

    def log_density_ratio(self,x):
        return self.logit_r(x).squeeze(-1)

    def train(self, epochs, batch_size = None, lr = 1e-3):
        self.para_list = list(self.parameters())

        self.optimizer = torch.optim.Adam(self.para_list, lr=lr)
        if batch_size is None:
            batch_size = self.label_1_samples.shape[0]
        dataset_0 = torch.utils.data.TensorDataset(self.label_0_samples)
        dataset_1 = torch.utils.data.TensorDataset(self.label_1_samples)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        pbar = tqdm(range(epochs))
        for t in pbar:
            dataloader_0 = torch.utils.data.DataLoader(dataset_0, batch_size=batch_size, shuffle=True)
            dataloader_1 = torch.utils.data.DataLoader(dataset_1, batch_size=batch_size, shuffle=True)
            for batch_0, batch_1 in zip(dataloader_0, dataloader_1):
                label_0_batch = batch_0[0]
                label_1_batch = batch_1[0]
                self.optimizer.zero_grad()
                batch_loss = self.loss(label_1_batch,label_0_batch)
                batch_loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor([self.loss(batch_1[0].to(device),batch_0[0].to(device)) for batch_0, batch_1 in zip(dataloader_0, dataloader_1)]).mean().item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)))
        self.to(torch.device('cpu'))
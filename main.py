import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter


class VAE(nn.Module):
    def __init__(self, img_dim, z_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # 编码器
        self.encoder_input = nn.Linear(img_dim, 256)
        self.encoder_hid = nn.Linear(256, 128)
        self.encoder_mean = nn.Linear(128, z_dim)
        self.encoder_var = nn.Linear(128, z_dim)
        # 解码器
        self.decoder_in = nn.Linear(z_dim, 256)
        self.decoder_hid = nn.Linear(256, 128)
        self.decoder_output = nn.Linear(128, img_dim)

    def encoder(self, x):
        h = self.relu(self.encoder_hid(self.encoder_input(x)))
        mean = self.encoder_mean(h)
        var = self.encoder_var(h)
        return mean, var

    def decoder(self, x):
        r = self.relu(self.decoder_hid(self.decoder_in(x)))
        return self.sigmoid(self.decoder_output(r))

    def forward(self, x):
        mean, var = self.encoder(x)
        e = torch.rand_like(var)
        z = mean + e*var
        recon_img = self.decoder(z)
        return recon_img, mean, var


device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 20
img_dim = 28 * 28 * 1
batch_size = 32
epochs = 20

VAE = VAE(img_dim, z_dim).to(device)

dataset = datasets.MNIST(root="dataset/", transform=transforms.ToTensor(), download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt = optim.Adam(VAE.parameters(), lr=lr)
criterion = nn.BCELoss(reduction="sum")
fake_img = SummaryWriter(f"log/fake/")
real_img = SummaryWriter(f"log/real/")
loss_img = SummaryWriter(f"log/loss/")
step = 0

for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 28 * 28).to(device)
        batch_size = real.shape[0]

        fake, mean, var = VAE(real)
        loss_recon = criterion(fake, real)
        loss_kl = -0.5 * torch.sum(1 + torch.log(var.pow(2)) - mean.pow(2) - var.pow(2))
        loss = loss_recon + loss_kl
        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}]     Loss: {loss:.4f}"
            )

            with torch.no_grad():
                fake = fake.reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                fake_img.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                real_img.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                loss_img.add_scalar("Loss", loss.item(), global_step=step)
                step += 1
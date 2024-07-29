import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

def generate_hq_image(generator, save_path):
    # Generate a single latent vector (usually from a normal distribution)
    latent_vector = torch.randn(1, 100, 1, 1)  # Assuming your latent space size is 100
    
    # Generate the fake image
    with torch.no_grad():
        fake_image = generator(latent_vector)
    
    # Normalize the image to [0, 1] range
    fake_image = (fake_image + 1) / 2.0
    
    # Save the generated image
    save_image(fake_image, save_path, normalize=True)

if __name__ == "__main__":
    # Load the generator model
    generator = Generator()
    generator.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
    for i in range(1000): 
        generator.eval()  # Set the generator to evaluation mode
        
        # Generate and save a high-quality image
        generate_hq_image(generator, f'hq_image{i}.png')

from torch.utils import data
import numpy as np
import  pytorch_fid_wrapper as pfw
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torchvision.utils import save_image
import math
from sklearn import preprocessing
from skimage import io
from IPython.display import clear_output
import pickle

class CustomDataset(data.Dataset):
    
    def __init__(self, dataset, transform=None):
        super().__init__()
        self.x = torch.from_numpy(dataset[0]).permute(0,3,1,2)
        self.y = torch.from_numpy(dataset[1])
        self.len = dataset[0].shape[0]
        self.transform = transform

    def __getitem__(self, index):
        image = self.x[index]
        label = self.y[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.len

def load_dataset(with_val=False):

    labels = os.listdir('small-wikiart')
    n_classes = len(labels)

    x_train = []
    y_train = []
    count = 0
    img_count = 0
    print('load started')
    for label in labels:
        for image_name in os.listdir('small-wikiart/{}'.format(label)):
            img = io.imread('small-wikiart/{}/{}'.format(label, image_name))
            img = (img-127.5)/127.5
            x_train.append(img)
            y_train.append(label)
            img_count+=1
        count+=1
        print('{}[{}]'.format(count/len(labels)*100, img_count))
    print('load finished :D')
    y_train = np.array(y_train).reshape(-1,1)
    x_train = np.array(x_train)

    enc = preprocessing.OneHotEncoder()
    enc.fit(y_train)
    y_train = enc.transform(y_train).toarray()

    return n_classes, x_train, y_train

def orthogonal_regularization(weight):
    '''
    Function for computing the orthogonal regularization term for a given weight matrix.
    '''
    weight = weight.flatten(1)
    return torch.norm(
        torch.dot(weight, weight) * (torch.ones_like(weight) - torch.eye(weight.shape[0]))
    )

def show_tensor_images(image_tensor, num_images=16, size=(3, 32, 32), nrow=4, show=True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()

class ClassConditionalBatchNorm2d(nn.Module):
    '''
    ClassConditionalBatchNorm2d Class
    Values:
    in_channels: the dimension of the class embedding (c) + noise vector (z), a scalar
    out_channels: the dimension of the activation tensor to be normalized, a scalar
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.class_scale_transform = nn.utils.spectral_norm(nn.Linear(in_channels, out_channels, bias=False))
        self.class_shift_transform = nn.utils.spectral_norm(nn.Linear(in_channels, out_channels, bias=False))

    def forward(self, x, y):
        normalized_image = self.bn(x)
        class_scale = (1 + self.class_scale_transform(y))[:, :, None, None]
        class_shift = self.class_shift_transform(y)[:, :, None, None]
        transformed_image = class_scale * normalized_image + class_shift
        return transformed_image

class AttentionBlock(nn.Module):
    '''
    AttentionBlock Class
    Values:
    channels: number of channels in input
    '''
    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        self.theta = nn.utils.spectral_norm(nn.Conv2d(channels, channels // 8, kernel_size=1, padding=0, bias=False))
        self.phi = nn.utils.spectral_norm(nn.Conv2d(channels, channels // 8, kernel_size=1, padding=0, bias=False))
        self.g = nn.utils.spectral_norm(nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=False))
        self.o = nn.utils.spectral_norm(nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=False))

        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        spatial_size = x.shape[2] * x.shape[3]

        # Apply convolutions to get query (theta), key (phi), and value (g) transforms
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), kernel_size=2)
        g = F.max_pool2d(self.g(x), kernel_size=2)

        # Reshape spatial size for self-attention
        theta = theta.view(-1, self.channels // 8, spatial_size)
        phi = phi.view(-1, self.channels // 8, spatial_size // 4)
        g = g.view(-1, self.channels // 2, spatial_size // 4)

        # Compute dot product attention with query (theta) and key (phi) matrices
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), dim=-1)

        # Compute scaled dot product attention with value (g) and attention (beta) matrices
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.channels // 2, x.shape[2], x.shape[3]))

        # Apply gain and residual
        return self.gamma * o + x

class GResidualBlock(nn.Module):
    '''
    GResidualBlock Class
    Values:
    c_dim: the dimension of conditional vector [c, z], a scalar
    in_channels: the number of channels in the input, a scalar
    out_channels: the number of channels in the output, a scalar
    '''

    def __init__(self, c_dim, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        self.bn1 = ClassConditionalBatchNorm2d(c_dim, in_channels)
        self.bn2 = ClassConditionalBatchNorm2d(c_dim, out_channels)

        self.activation = nn.ReLU()
        self.upsample_fn = nn.Upsample(scale_factor=2)     # upsample occurs in every gblock

        self.mixin = (in_channels != out_channels)
        if self.mixin:
            self.conv_mixin = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))

    def forward(self, x, y):
        # h := upsample(x, y)
        h = self.bn1(x, y)
        h = self.activation(h)
        h = self.upsample_fn(h)
        h = self.conv1(h)

        # h := conv(h, y)
        h = self.bn2(h, y)
        h = self.activation(h)
        h = self.conv2(h)

        # x := upsample(x)
        x = self.upsample_fn(x)
        if self.mixin:
            x = self.conv_mixin(x)

        return h + x

class Generator(nn.Module):
    '''
    Generator Class
    Values:
    z_dim: the dimension of random noise sampled, a scalar
    shared_dim: the dimension of shared class embeddings, a scalar
    base_channels: the number of base channels, a scalar
    bottom_width: the height/width of image before it gets upsampled, a scalar
    n_classes: the number of image classes, a scalar
    '''

    def __init__(self, base_channels=96, bottom_width=4, z_dim=120, shared_dim=128, n_classes=1000):
        super().__init__()

        n_chunks = 6    # 5 (generator blocks) + 1 (generator input)
        self.z_chunk_size = z_dim // n_chunks
        self.z_dim = z_dim
        self.shared_dim = shared_dim
        self.bottom_width = bottom_width

        # No spectral normalization on embeddings, which authors observe to cripple the generator
        self.shared_emb = nn.Embedding(n_classes, shared_dim)

        self.proj_z = nn.Linear(self.z_chunk_size, 8 * base_channels * bottom_width ** 2)

        # Can't use one big nn.Sequential since we are adding class+noise at each block
        self.g_blocks = nn.ModuleList([
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 8 * base_channels, 4 * base_channels),
                AttentionBlock(4 * base_channels),
            ]),
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 4 * base_channels, 2 * base_channels),
                AttentionBlock(2 * base_channels),
            ]),
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 2*base_channels, base_channels),
                AttentionBlock(base_channels),
            ]),
        ])
        self.proj_o = nn.Sequential(
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(base_channels, 3, kernel_size=1, padding=0)),
            nn.Tanh(),
        )

    def forward(self, z, y):
        '''
        z: random noise with size self.z_dim
        y: class embeddings with size self.shared_dim
            = NOTE =
            y should be class embeddings from self.shared_emb, not the raw class labels
        '''
        # Chunk z and concatenate to shared class embeddings
        zs = torch.split(z, self.z_chunk_size, dim=1)
        z = zs[0]
        ys = [torch.cat([y, z], dim=1) for z in zs[1:]]

        # Project noise and reshape to feed through generator blocks
        h = self.proj_z(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Feed through generator blocks
        for idx, g_block in enumerate(self.g_blocks):
            h = g_block[0](h, ys[idx])
            h = g_block[1](h)

        # Project to 3 RGB channels with tanh to map values to [-1, 1]
        h = self.proj_o(h)

        return h

    def loss(self, dis_fake):        
        loss = -torch.mean(dis_fake)
        return loss

class DResidualBlock(nn.Module):
    '''
    DResidualBlock Class
    Values:
    in_channels: the number of channels in the input, a scalar
    out_channels: the number of channels in the output, a scalar
    downsample: whether to apply downsampling
    use_preactivation: whether to apply an activation function before the first convolution
    '''

    def __init__(self, in_channels, out_channels, downsample=True, use_preactivation=False):
        super().__init__()

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        self.activation = nn.ReLU()
        self.use_preactivation = use_preactivation  # apply preactivation in all except first dblock

        self.downsample = downsample    # downsample occurs in all except last dblock
        if downsample:
            self.downsample_fn = nn.AvgPool2d(2)
        self.mixin = (in_channels != out_channels) or downsample
        if self.mixin:
            self.conv_mixin = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))

    def _residual(self, x):
        if self.use_preactivation:
            if self.mixin:
                x = self.conv_mixin(x)
            if self.downsample:
                x = self.downsample_fn(x)
        else:
            if self.downsample:
                x = self.downsample_fn(x)
            if self.mixin:
                x = self.conv_mixin(x)
        return x

    def forward(self, x):
        # Apply preactivation if applicable
        if self.use_preactivation:
            h = F.relu(x)
        else:
            h = x

        h = self.conv1(h)
        h = self.activation(h)
        if self.downsample:
            h = self.downsample_fn(h)

        return h + self._residual(x)

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
    base_channels: the number of base channels, a scalar
    n_classes: the number of image classes, a scalar
    '''

    def __init__(self, base_channels=96, n_classes=1000):
        super().__init__()

        # For adding class-conditional evidence
        self.shared_emb = nn.utils.spectral_norm(nn.Embedding(n_classes, 8 * base_channels))

        self.d_blocks = nn.Sequential(
            DResidualBlock(3, base_channels, downsample=True, use_preactivation=False),
            AttentionBlock(base_channels),

            DResidualBlock(base_channels, 2 * base_channels, downsample=True, use_preactivation=True),
            AttentionBlock(2 * base_channels),

            DResidualBlock(2 * base_channels, 4 * base_channels, downsample=True, use_preactivation=True),
            AttentionBlock(4 * base_channels),

            DResidualBlock(4 * base_channels, 8 * base_channels, downsample=False, use_preactivation=True),
            AttentionBlock(8 * base_channels),

            nn.ReLU(inplace=True),
        )
        self.proj_o = nn.utils.spectral_norm(nn.Linear(8 * base_channels, 1))

    def forward(self, x, y=None):
        h = self.d_blocks(x)
        h = torch.sum(h, dim=[2, 3])

        # Class-unconditional output
        uncond_out = self.proj_o(h)
        if y is None:
            return uncond_out

        # Class-conditional output
        cond_out = torch.sum(self.shared_emb(y) * h, dim=1, keepdim=True)
        return uncond_out + cond_out

    def loss(self, dis_fake, dis_real):
        loss = torch.mean(F.relu(1. - dis_real))
        loss += torch.mean(F.relu(1. + dis_fake))
        return loss

class RandomApplyEach(nn.Module):
    def __init__(self, transforms, p):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, img):
        for t in self.transforms:
            if self.p > torch.rand(1, device='cuda'):
                img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ImageDatasetWrapper():
    def __init__(self, dataset):
        self.dataset=dataset

    def __getitem__(self, key):
        if isinstance(key, slice):
            range(*key.indices(len(self.dataset)))
            return torch.tensor([np.asarray(self.dataset[i][0]) for i in range(*key.indices(len(self.dataset)))])
        elif isinstance(key, int):
            return torch.tensor(self.dataset[key][0])

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    device = 'cuda'
    # load data
    # n_classes, x_train, y_train = load_dataset()

    model_dir = './generators_weights/'
    img_dir = './generated_images/'
    var_dir = './saved_variables/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(var_dir):
        os.makedirs(var_dir)

    batch_size = 16
    dims = [64, 64]
    transforms_compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(dims)])
    # dataset = CustomDataset((x_train,y_train), transform=transforms_compose)
    print("Creating dataset object")
    dataset = datasets.ImageFolder(root='small-wikiart/', transform=transforms_compose)
    n_classes = len(dataset.classes)
    loader = data.DataLoader(dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)

    augmentation_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=90, translate=(0.3, 0.3), scale=(0.5, 1.5), fill=0),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ]
    p = torch.tensor(0.0, device=device)
    ada_target = 0.6
    update_iteration = 8
    adjustment_size = 500000 # number of images to reach p=1
    augmentation = RandomApplyEach(augmentation_transforms, p).to(device)
    ada_buf = torch.tensor([0.0, 0.0], device=device)

    # Initialize models
    print("Creating models")
    base_channels = 64
    z_dim = 120
    shared_dim = 128
    generator = Generator(base_channels=base_channels, bottom_width=8, z_dim=z_dim, shared_dim=shared_dim, n_classes=n_classes).to(device)
    discriminator = Discriminator(base_channels=base_channels, n_classes=n_classes).to(device)
    
    # Initialize weights orthogonally
    for module in generator.modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
            nn.init.orthogonal_(module.weight)
    for module in discriminator.modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
            nn.init.orthogonal_(module.weight)

    # Initialize optimizers
    gen_opt = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.999), eps=1e-6)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=4e-4, betas=(0.0, 0.999), eps=1e-6)

    # Setup FID
    print("Calculating FID parameters")
    pfw.set_config(batch_size=batch_size, device=device)
    if os.path.isfile(var_dir+'fid_stats.pkl'):
        with open(var_dir+'fid_stats.pkl', 'rb') as f:
            real_m, real_s = pickle.load(f)        
    else:
        real_m, real_s = pfw.get_stats(ImageDatasetWrapper(dataset))
        with open(var_dir+'fid_stats.pkl', 'wb') as f:
            pickle.dump([real_m, real_s], f)
    print("FID parameters calculated!")
    
    ###########
    #  TRAIN  #
    ###########
    print("Training started")
    cur_step = 0
    min_fid = 2000
    e = 0
    D_steps = 2
    n_epochs = 100
    
    fixed_z = torch.randn(batch_size, z_dim, device=device)       # Generate random noise (z)
    fixed_image, fixed_y = next(iter(loader))
    save_image(fixed_image, img_dir+"real.png")
    fixed_y = fixed_y.to(device).long()                               # Generate a batch of labels (y), one for each class
    fixed_y_emb = generator.shared_emb(fixed_y)                         # Retrieve class embeddings (y_emb) from generator

    image_number = 0

    fakes = torch.tensor([], device='cpu')
    fids = torch.tensor([], device='cpu')
    
    for epoch in range(n_epochs):
        print('##############################')
        print('#epoch: {}'.format(epoch))

        for batch_ndx, sample in enumerate(loader):
            real, labels = sample[0], sample[1]
            batch_size = len(real)
            real = real.to(device)
            real_augmented = augmentation(real)

            disc_opt.zero_grad()
            gen_opt.zero_grad()

            for i in range(D_steps):
                # Zero out the discriminator gradients
                disc_opt.zero_grad()
                ### Update discriminator ###
                # Get noise corresponding to the current batch_size 
                z = torch.randn(batch_size, z_dim, device=device)       # Generate random noise (z)
                y = labels.to(device).long()                            # Generate a batch of labels (y), one for each class
                y_emb = generator.shared_emb(y)                         # Retrieve class embeddings (y_emb) from generator
                fake = generator(z, y_emb)
                fake = augmentation(fake.detach())

                disc_fake_pred = discriminator(fake, y)
                disc_real_pred = discriminator(real_augmented, y)

                ada_buf += torch.tensor(
                    (torch.clamp(torch.sign(disc_real_pred), min=0, max=1).sum().item(), disc_real_pred.shape[0]),
                    device=device
                )

                #loss
                disc_loss = discriminator.loss(disc_fake_pred, disc_real_pred)
                # Update gradients
                disc_loss.backward()
                # Update optimizer
                disc_opt.step()

            ### Update generator ###
            # Zero out the generator gradients
            gen_opt.zero_grad()

            fake = generator(z, y_emb)
            fake = augmentation(fake)
            disc_fake_pred = discriminator(fake, y)
            #loss
            gen_loss =  generator.loss(disc_fake_pred)
            # Update gradients
            gen_loss.backward()
            # Update optimizer
            gen_opt.step()

            fakes = torch.cat((fakes, fake.to('cpu')))

            if cur_step % update_iteration == 0:
                # Adaptive Data Augmentation
                pred_signs, n_pred = ada_buf
                r_t = pred_signs / n_pred

                sign = r_t - ada_target

                augmentation.p = torch.clamp(augmentation.p + (sign * n_pred / adjustment_size), min=0, max=1)

                ada_buf = ada_buf * 0

            cur_step +=1

            if cur_step % 500 == 0:
                print('===========================================================================')
                val_fid = pfw.fid(fakes, real_m=real_m, real_s=real_s)
                fids = torch.cat((fids, torch.tensor([val_fid])))
                with open(var_dir+'fids.pkl', 'wb') as f:
                    pickle.dump(fids, f)
                with open(var_dir+'p.pkl', 'wb') as f:
                    pickle.dump(augmentation.p, f)
                fakes = torch.tensor([], device='cpu')
                print('FID: {}'.format(val_fid))
                print('augmentation p: {}'.format(augmentation.p))
                if (val_fid < min_fid):
                    min_fid = val_fid
                    save_image(fake, img_dir+"generated-with-better-FID{}.png".format(image_number))
                    torch.save(generator.state_dict(), (model_dir+'gen.state_dict'))
                    torch.save(discriminator.state_dict(), (model_dir+'disc.state_dict'))
                print('===========================================================================')

        print('saved images')
        fake = generator(fixed_z, fixed_y_emb)
        save_image(fake, img_dir+"generated{}.png".format(image_number))
        save_image(augmentation(fixed_image), img_dir+"augmented_real.png")
        image_number += 1

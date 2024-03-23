import torch

class Resnet_SISR(torch.nn.Module):
    def __init__(self, residual_depth, in_channels=1, maginfication=2, num_magnifications=1, latent_channel_count=64):
        super(Resnet_SISR, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels,out_channels=latent_channel_count,kernel_size=9,padding=4)
        self.relu = torch.nn.ReLU()
        self.bn_res = torch.nn.BatchNorm2d(latent_channel_count)
        self.residual_depth = residual_depth
        self.residual_layer_list = torch.nn.ModuleList()
        self.magnification_list = torch.nn.ModuleList()
        self.magnification = maginfication
        self.num_magnifications = num_magnifications
        for _ in range(residual_depth):
            self.residual_layer_list.append(self.make_residual_block(latent_channel_count))
        self.conv2 = torch.nn.Conv2d(in_channels=latent_channel_count,out_channels=latent_channel_count,kernel_size=9,padding=4)
        for _ in range(num_magnifications):
            self.magnification_list.append(self.make_subpixel_block(latent_channel_count))
        self.conv3 = torch.nn.Conv2d(in_channels=latent_channel_count,out_channels=in_channels,kernel_size=3,padding=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        res_0 = x.clone()
        for i in range(self.residual_depth):
            res = x.clone()
            x = self.residual_layer_list[i](x)
            x = x + res
        x = self.conv2(x)
        x = self.bn_res(x)
        x = x + res_0
        for i in range(self.num_magnifications):
            x = self.magnification_list[i](x)
        x = self.conv3(x)
        x = self.relu(x)
        return x

    def make_residual_block(self, channels):
        return torch.nn.Sequential(torch.nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1),
                                   torch.nn.BatchNorm2d(channels),
                                   self.relu,
                                   torch.nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1),
                                   torch.nn.BatchNorm2d(channels),
                                   self.relu)
    def make_subpixel_block(self, channels):
        return torch.nn.Sequential(torch.nn.Conv2d(in_channels=channels,out_channels=channels*self.magnification*self.magnification,kernel_size=3,padding=1),
                                   torch.nn.PixelShuffle(self.magnification),
                                   self.relu)
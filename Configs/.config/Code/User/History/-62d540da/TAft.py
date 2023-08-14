import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(m):
    """
    Glorot uniform initialization for network.
    """
    if 'conv' in m.__class__.__name__.lower():
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
        m.bias.data.fill_(0.01)

class padding(nn.Module):
    def __init__(self):
        super(padding,self).__init__()
        self.wpad = nn.ReplicationPad3d((0, -1, 0, 0,0,0))
        self.hpad = nn.ReplicationPad3d((0, 0, 0, -1,0,0))
        self.dpad = nn.ReplicationPad3d(((0,0,0,0,0,-1)))
    def forward(self, input, targetsize):
        if input.size()[2] != targetsize[2]:
            input = self.dpad(input)
        if input.size()[3] != targetsize[3]:
            input = self.hpad(input)
        if input.size()[4] != targetsize[4]:
            input = self.wpad(input)
        return input
# Based on FlowNet2S: https://arxiv.org/abs/1612.01925
# https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py
'''
Portions of this code copyright 2017, Clement Pinard
'''
def conv(in_channels, out_channels, kernel_size=3, stride=1, latent=False):
    
    if latent:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=False),
            nn.ReLU(inplace=True))

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose3d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

def predict_flow(in_planes):
    return nn.Conv3d(in_planes,3,kernel_size=3,stride=1,padding=1,bias=True)

# from FlowNet_PET
def upconv(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.LeakyReLU(inplace=True))
# from FlowNet_PET
def concatenate(tensor1, tensor2, tensor3):
    _, _, d1, h1, w1 = tensor1.shape
    _, _, d2, h2, w2 = tensor2.shape
    _, _, d3, h3, w3 = tensor3.shape
    d, h, w = min(d1, d2, d3), min(h1, h2, h3), min(w1, w2, w3)
    return torch.cat((tensor1[:, :, :d, :h, :w], tensor2[:, :, :d, :h, :w], tensor3[:, :, :d, :h, :w]), 1)

class net3d(nn.Module):
    def __init__(self):
        super(net3d, self).__init__()
        print("Building model...")
        # padding for input data: divisible by 64
        self.pad = padding()
        self.conv1   = conv( 2,   4, kernel_size=3, stride=2)
        self.conv2   = conv( 4,  8, kernel_size=3, stride=2)
        self.conv3   = conv(8,  16, kernel_size=3, stride=2)

        self.conv_latent = conv(16, 32, kernel_size=3, stride=1, latent=True)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

       # self.deconv3 = deconv(512,128)
      #  self.deconv2 = deconv(386,64)

        self.predict_flow_latent = predict_flow(32)
        self.predict_flow3 = predict_flow(26)
        self.predict_flow2 = predict_flow(14)
        self.predict_flow1 = predict_flow(8)

        # upsample feature maps
        self.upconv_latent = upconv(32, 16)
        self.upconv3 = upconv(26, 8)
        self.upconv2 = upconv(14, 4)

        self.upconvflow_latent = nn.ConvTranspose3d(3, 2, 4, 2, 1, bias=False)
        self.upconvflow3 = nn.ConvTranspose3d(3, 2, 4, 2, 1, bias=False)
        self.upconvflow2 = nn.ConvTranspose3d(3, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose3d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        # init_deconv_bilinear(m.weight)
       # self.upsample1 = nn.Upsample(scale_factor=4, mode='trilinear',align_corners=True)


    def forward(self, x):
        out_conv1 = self.conv1(x)
     #   out_conv1 = self.dropout(out_conv1)
        out_conv2 = self.conv2(out_conv1)
    #    out_conv2 = self.dropout(out_conv2)
        out_conv3 = self.conv3(out_conv2)
        out_conv3 = self.dropout(out_conv3)
        out_conv_latent = self.conv_latent(out_conv3)
        out_upconv_latent = self.upconv_latent(out_conv_latent)

        flow_latent = self.predict_flow_latent(out_conv_latent)
        up_flow_latent = self.upconvflow_latent(flow_latent)


        # Combine upsampled features with intermediate downsampled features and current flow
        concat3 = concatenate(out_upconv_latent, out_conv2, up_flow_latent)
        # Predict the next flow with better resolution
        flow3 = self.predict_flow3(concat3)
        # Upsample flow to higher resolution
        up_flow3 = self.upconvflow3(flow3)
        # Upsample features
        out_upconv3 = self.upconv3(concat3)

        # Combine upsampled features with intermediate downsampled features and current flow
        concat2 = concatenate(out_upconv3, out_conv1, up_flow3)

        # Predict the next flow with better resolution
        flow2 = self.predict_flow2(concat2)
        # Upsample flow to higher resolution
        up_flow2 = self.upconvflow2(flow2)
        # Upsample features
        out_upconv2 = self.upconv2(concat2)
    
        # Combine upsampled features with intermediate downsampled features and current flow
        concat1 = concatenate(out_upconv2, x, up_flow2)

         # Predict the next flow at original resolution
        flow1 = self.predict_flow1(concat1)

         # Free up memory
        del (out_conv1, out_conv2, out_conv_latent, up_flow_latent, out_upconv_latent,
             concat3, up_flow3, out_upconv3, concat2, up_flow2, out_upconv2, concat1)
        

        return flow1, flow2, flow3, flow_latent

def gaussian_kernel(kernel_size=3, sigma=0.1, dim=3, channels=1):
    
    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
    
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * np.sqrt(2 * np.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    return kernel        

def generate_grid(D, H, W, device):
    
    # Create meshgrid of the voxel locations
    z_grid, y_grid, x_grid = torch.meshgrid(torch.arange(0,D),
                                            torch.arange(0,H),
                                            torch.arange(0,W))
    grid = torch.stack((x_grid, y_grid, z_grid), 3).float()
    
    # Scale between -1 and 1
    grid = 2*grid / (torch.tensor([W, H, D])-1) - 1
    
    return grid.to(device)


class FlowNetPET(nn.Module):
    def __init__(self, architecture_config, device):
        super(FlowNetPET, self).__init__()
        
        # Read configuration
        self.input_shape = eval(architecture_config['input_shape'])
        self.interp_mode = architecture_config['interp_mode']
                
        # Gaussian kernel for blurring
        self.gauss_kernel_len = int(architecture_config['gauss_kernel_len'])
        if self.gauss_kernel_len>0:
            self.gauss_kernel = gaussian_kernel(self.gauss_kernel_len, 
                                                float(architecture_config['gauss_sigma'])).to(device)
        
        # Create FlowNet
        self.predictor = net3d().to(device)
        
        # Create grid for every flow resolution
        flows = self.predictor(torch.ones(1,2,*self.input_shape).to(device))
        self.grids = []
        print('The flow predictions will have sizes:')
        for flow in flows:
            b,_,d,h,w = flow.shape
            print('%i x %i x %i' % (d,h,w))
            self.grids.append(generate_grid(d, h, w, device))

    def warp_frame(self, flow, frame, grid=None, interp_mode='bilinear'):
        if grid is None:
            grid = self.grids[0]
                       
        warped_frame = F.grid_sample(frame, grid+flow.permute(0,2,3,4,1), 
                                     mode=interp_mode, padding_mode='border', align_corners=True)

        return warped_frame
    
    def apply_shift(self, flow, frame, grid):

        b, _, d, h, w = flow.shape
        if ((w==frame.shape[-1]) & (self.interp_mode=='nearest')):
            # Use gradients from bilinear but the data from nearest to allow backprop
            warped_frame = self.warp_frame(flow, frame, grid, interp_mode='bilinear')
            warped_frame.data = self.warp_frame(flow, frame, grid, interp_mode='nearest')
        else:
            frame = F.interpolate(frame, size=(d, h, w), mode='trilinear', align_corners=True)
            warped_frame = self.warp_frame(flow, frame, grid)

        return warped_frame

    def gaussian_blur(self, img):
        padding = int((self.gauss_kernel_len - 1) / 2)
        img = torch.nn.functional.pad(img, (padding, padding, padding, padding, padding, padding), mode='replicate')
        return torch.nn.functional.conv3d(img, self.gauss_kernel, groups=1)
    
    def forward(self, x):
        
        # Predict flows from two frames
        flow_predictions = self.predictor(x)
        
        # Apply flow to first frame at each resolutions
        warped_images = [self.apply_shift(flow, 
                                          self.gaussian_blur(x[:, :1, :, :]), 
                                          grid) for flow, grid in zip(flow_predictions, self.grids)]
        
        return flow_predictions, warped_images
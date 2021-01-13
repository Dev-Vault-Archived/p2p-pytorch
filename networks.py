import torch
import torch.nn as nn
import numpy as np
from math import pi
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import functools
# import random
from torch.optim import lr_scheduler

import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import Parameter

import random

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images

# def get_norm_layer(norm_type='instance'):
#     if norm_type == 'batch':
#         norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
#     elif norm_type == 'instance':
#         norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
#     elif norm_type == 'none':
#         norm_layer = None
#     else:
#         raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
#     return norm_layer

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net.to(gpu_id)
    init_weights(net, init_type, gain=init_gain)
    return net

def define_C(init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = None

    net = CompressionNetwork()

    return init_net(net, init_type, init_gain, gpu_id)

def define_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = None

    # norm_layer = get_norm_layer(norm_type='batch')
    # net = ResnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)
    net = ExpandNetwork()

    return init_net(net, init_type, init_gain, gpu_id)

class CompressionNetwork(nn.Module):
    def __init__(self):
        super(CompressionNetwork, self).__init__()

        self.conv_input = nn.Sequential(
            ConvLayer(3, 64, kernel_size=2, stride=1),
            nn.LeakyReLU(0.2)
        )

        self.conv_block1 = nn.Sequential(
            ConvLayer(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.conv_block2 = nn.Sequential(
            ConvLayer(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.conv_block3 = nn.Sequential(
            ConvLayer(64, 12, kernel_size=3, stride=2),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2)
        )

        self.pooling = nn.AvgPool2d(2, stride=2)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        identity = x
        res = self.conv_input(x)
        res = self.conv_block1(res)
        res = self.conv_block2(res)
        res = self.conv_block3(res)
        # res = self.conv_block4(res)
        # res = self.conv_block5(res)
        print(res.size())

        res = self.pooling(res)
        print(res.size())

        res = self.shuffle(res)
        print(res.size())
        
        res = F.normalize(res, p=2, dim=1)
        print(res.size())

        res = F.interpolate(res, scale_factor=2)
        print(res.size())

        return identity + res

# class CompressionEncoder(nn.Module):
#     def __init__(self):
#         super(CompressionEncoder, self).__init__()

#         self.conv_block1 = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(3, 60, kernel_size=(7, 7), stride=1),
#             nn.InstanceNorm2d(60, eps=1e-3, momentum=0.1, affine=True, track_running_stats=False),
#             nn.ReLU()
#         )

#         self.conv_block2 = nn.Sequential(
#             nn.ReflectionPad2d((0,1,1,0)),
#             nn.Conv2d(60, 120, 3, stride=2, padding=0, padding_mode='reflect'),
#             nn.InstanceNorm2d(120, eps=1e-3, momentum=0.1, affine=True, track_running_stats=False),
#             nn.ReLU()
#         )

#         self.conv_block3 = nn.Sequential(
#             nn.ReflectionPad2d((0,1,1,0)),
#             nn.Conv2d(120, 240, 3, stride=2, padding=0, padding_mode='reflect'),
#             nn.InstanceNorm2d(240, eps=1e-3, momentum=0.1, affine=True, track_running_stats=False),
#             nn.ReLU()
#         )

#         self.conv_block4 = nn.Sequential(
#             nn.ReflectionPad2d((0,1,1,0)),
#             nn.Conv2d(240, 480, 3, stride=2, padding=0, padding_mode='reflect'),
#             nn.InstanceNorm2d(480, eps=1e-3, momentum=0.1, affine=True, track_running_stats=False),
#             nn.ReLU()
#         )

#         self.conv_block5 = nn.Sequential(
#             nn.ReflectionPad2d((0,1,1,0)),
#             nn.Conv2d(480, 960, 3, stride=2, padding=0, padding_mode='reflect'),
#             nn.InstanceNorm2d(960, eps=1e-3, momentum=0.1, affine=True),
#             nn.ReLU()
#         )

#         self.conv_block_out = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(960, 220, 3, stride=1)
#         )

#     def forward(self, x):
#         x = self.conv_block1(x)
#         x = self.conv_block2(x)
#         x = self.conv_block3(x)
#         x = self.conv_block4(x)
#         x = self.conv_block5(x)
#         out = self.conv_block_out(x)
#         return out

# class CompressionResidualBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(CompressionResidualBlock, self).__init__()

#         self.activation = nn.ReLU()

#         self.interlayer_norm = nn.InstanceNorm2d()

#         pad_size = int((3-1)/2)
#         self.pad = nn.ReflectionPad2d(pad_size)
#         self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride=1)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, 3, stride=1)
#         self.norm1 = self.interlayer_norm(in_channels, momentum=0.1, affine=True, track_running_stats=False)
#         self.norm2 = self.interlayer_norm(in_channels, momentum=0.1, affine=True, track_running_stats=False)

#     def forward(self, x):
#         identity_map = x
#         res = self.pad(x)
#         res = self.conv1(res)
#         res = self.norm1(res) 
#         res = self.activation(res)

#         res = self.pad(res)
#         res = self.conv2(res)
#         res = self.norm2(res)

#         return torch.add(res, identity_map)

# class CompressionGenerator(nn.Module):
#     def __init__(self):
#         super(CompressionGenerator, self).__init__()

#         self.conv_block_init = nn.Sequential(
#             nn.InstanceNorm2d(220, eps=1e-3, momentum=0.1, affine=True),
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(220, 960, kernel_size=(3, 3), stride=1),
#             nn.InstanceNorm2d(960, eps=1e-3, momentum=0.1, affine=True)
#         )

#         for m in range(8):
#             resblock_m = CompressionResidualBlock(960)
#             self.add_module(f'resblock_{str(m)}', resblock_m)

#         self.upconv_block1 = nn.Sequential(
#             nn.ConvTranspose2d(960, 480, 3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm2d(480, eps=1e-3, momentum=0.1, affine=True),
#             nn.ReLU()
#         )

#         self.upconv_block2 = nn.Sequential(
#             nn.ConvTranspose2d(480, 240, 3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm2d(240, eps=1e-3, momentum=0.1, affine=True),
#             nn.ReLU()
#         )

#         self.upconv_block3 = nn.Sequential(
#             nn.ConvTranspose2d(240, 120, 3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm2d(120, eps=1e-3, momentum=0.1, affine=True),
#             nn.ReLU()
#         )

#         self.upconv_block4 = nn.Sequential(
#             nn.ConvTranspose2d(120, 60, 3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm2d(60, eps=1e-3, momentum=0.1, affine=True),
#             nn.ReLU()
#         )

#         self.conv_block_out = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(60, 3, kernel_size=(7, 7), stride=1)
#         )

#     def forward(self, x):
        
#         head = self.conv_block_init(x)

#         for m in range(self.n_residual_blocks):
#             resblock_m = getattr(self, f'resblock_{str(m)}')
#             if m == 0:
#                 x = resblock_m(head)
#             else:
#                 x = resblock_m(x)
        
#         x += head
#         x = self.upconv_block1(x)
#         x = self.upconv_block2(x)
#         x = self.upconv_block3(x)
#         x = self.upconv_block4(x)
#         out = self.conv_block_out(x)

#         return out

# class CompressNetwork(nn.Module):
#     def __init__(self):
#         self.entropy_code = True

#         self.Encoder = CompressionEncoder()
#         self.Generator = CompressionGenerator()


#     def forward():
#         pass

# Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride) #, padding)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# Residual Block
#   adapted from pytorch tutorial
#   https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-
#   intermediate/deep_residual_network/main.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.BatchNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.BatchNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out 

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''

        return pixel_unshuffle(input, self.downscale_factor)

# Image Transform Network
class ExpandNetwork(nn.Module):
    def __init__(self):
        super(ExpandNetwork, self).__init__()
        
        # nonlineraity
        self.relu = nn.PReLU()
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

        # encoding layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1_e = nn.BatchNorm2d(32, affine=True)

        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2_e = nn.BatchNorm2d(64, affine=True)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3_e = nn.BatchNorm2d(128, affine=True)

        self.conv4 = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.in4_e = nn.BatchNorm2d(256, affine=True)

        self.pixel = PixelUnshuffle(2)
        self.conv4 = ConvLayer(128, 128, kernel_size=3, stride=1)

        # self.conv4 = ConvLayer(128, 128, kernel_size=3, stride=1)
        
        # residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.res6 = ResidualBlock(128)
        self.res7 = ResidualBlock(128)
        self.res8 = ResidualBlock(128)
        self.res9 = ResidualBlock(128)
        
        self.deconv4 = ConvLayer(128, 128, kernel_size=3, stride=1)

        # decoding layers
        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2 )
        self.in3_d = nn.BatchNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2 )
        self.in2_d = nn.BatchNorm2d(32, affine=True)

        self.deconv1 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1)
        self.in1_d = nn.BatchNorm2d(3, affine=True)

    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))
        y = self.relu(self.in4_e(self.conv4(y)))

        y = self.pixel(y)
        y = self.leakyRelu(self.conv4(y))

        # residual layers
        residual = y
        res = self.res1(y)
        res = self.res2(res)
        res = self.res3(res)
        res = self.res4(res)
        res = self.res5(res)
        res = self.res6(res)
        res = self.res7(res)
        res = self.res8(res)
        res = self.res9(res)

        res = res + residual
        y = self.leakyRelu(res)

        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.relu(self.in2_d(self.deconv2(y)))
        y = self.tanh(self.in1_d(self.deconv1(y)))
        # y = self.deconv1(y)

        return y

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

# class ExpandNetwork(nn.Module):    
#     def __init__(self):        
#         super(ExpandNetwork, self).__init__()        
        
#         self.layers = nn.Sequential(            
#             ConvLayer(3, 32, 9, 1),
#             ConvLayer(32, 64, 3, 2),
#             ConvLayer(64, 128, 3, 2),
            
#             ResidualLayer(128, 128, 3, 1),
#             ResidualLayer(128, 128, 3, 1),
#             ResidualLayer(128, 128, 3, 1),
#             ResidualLayer(128, 128, 3, 1),
#             ResidualLayer(128, 128, 3, 1),
#             ResidualLayer(128, 128, 3, 1),
#             ResidualLayer(128, 128, 3, 1),
#             ResidualLayer(128, 128, 3, 1),
#             ResidualLayer(128, 128, 3, 1),
            
#             DeconvLayer(128, 64, 3, 1),
#             DeconvLayer(64, 32, 3, 1),
#             ConvLayer(32, 3, 9, 1, activation='tanh'))
        
#     def forward(self, x):
#         return self.layers(x)

# class ConvLayer(nn.Module):    
#     def __init__(self, in_ch, out_ch, kernel_size, stride, pad='reflect', activation='relu', normalization='batch'):        
#         super(ConvLayer, self).__init__()
        
#         # padding
#         if pad == 'reflect':            
#             self.pad = nn.ReflectionPad2d(kernel_size//2)
#         elif pad == 'zero':
#             self.pad = nn.ZeroPad2d(kernel_size//2)
#         else:
#             raise NotImplementedError("Not expected pad flag !!!")
            
#         # convolution
#         self.conv_layer = nn.Conv2d(in_ch, out_ch, 
#                                     kernel_size=kernel_size,
#                                     stride=stride)
        
#         # activation
#         if activation == 'relu':
#             self.activation = nn.ReLU()        
#         elif activation == 'linear':
#             self.activation = lambda x : x
#         elif activation == 'tanh':
#             self.activation = nn.Tanh(True)
#         else:
#             raise NotImplementedError("Not expected activation flag !!!")

#         # normalization 
#         if normalization == 'instance':            
#             self.normalization = nn.InstanceNorm2d(out_ch, affine=True)
#         elif normalization == 'batch':
#             self.normalization = nn.BatchNorm2d(out_ch, affine=True)
#         else:
#             raise NotImplementedError("Not expected normalization flag !!!")

#     def forward(self, x):
#         x = self.pad(x)
#         x = self.conv_layer(x)
#         x = self.normalization(x)
#         x = self.activation(x)        
#         return x

# class ResidualLayer(nn.Module):    
#     def __init__(self, in_ch, out_ch, kernel_size, stride, pad='reflect', normalization='batch'):        
#         super(ResidualLayer, self).__init__()
        
#         self.conv1 = ConvLayer(in_ch, out_ch, kernel_size, stride, pad, 
#                                activation='relu', 
#                                normalization=normalization)
        
#         self.conv2 = ConvLayer(out_ch, out_ch, kernel_size, stride, pad, 
#                                activation='linear', 
#                                normalization=normalization)
        
#     def forward(self, x):
#         y = self.conv1(x)
#         return self.conv2(y) + x
        
# class DeconvLayer(nn.Module):    
#     def __init__(self, in_ch, out_ch, kernel_size, stride, pad='reflect', activation='relu', normalization='batch', upsample='nearest'):        
#         super(DeconvLayer, self).__init__()
        
#         # upsample
#         self.upsample = upsample
        
#         # pad
#         if pad == 'reflect':            
#             self.pad = nn.ReflectionPad2d(kernel_size//2)
#         elif pad == 'zero':
#             self.pad = nn.ZeroPad2d(kernel_size//2)
#         else:
#             raise NotImplementedError("Not expected pad flag !!!")        
        
#         # conv
#         self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride)
        
#         # activation
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         else:
#             raise NotImplementedError("Not expected activation flag !!!")
        
#         # normalization
#         if normalization == 'instance':
#             self.normalization = nn.InstanceNorm2d(out_ch, affine=True)
#         elif normalization == 'batch':
#             self.normalization = nn.BatchNorm2d(out_ch, affine=True)
#         else:
#             raise NotImplementedError("Not expected normalization flag !!!")
        
#     def forward(self, x):
#         x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample)        
#         x = self.pad(x)
#         x = self.conv(x)
#         x = self.normalization(x)        
#         x = self.activation(x)        
#         return x

def define_D(input_nc, ndf, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = None
    # norm_layer = get_norm_layer(norm_type=norm)

    net = MultiscaleDiscriminator(input_nc, ndf, n_layers=3, norm_layer=None, use_sigmoid=use_sigmoid, num_D=3, getIntermFeat=True)

    return init_net(net, init_type, init_gain, gpu_id)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=None, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=None, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

def sobelLayer(img, gpu_id='cuda:0'):
    img = img.squeeze(0)
    ten = torch.unbind(img)
    x = ten[0].unsqueeze(0).unsqueeze(0)
    
    a = np.array([[1, 0, -1], [2,0,-2], [1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).to(gpu_id)
    conv1.weight = nn.Parameter(torch.from_numpy(a).type(torch.cuda.FloatTensor).unsqueeze(0).unsqueeze(0))
    G_x = conv1(Variable(x)).data.view(1, x.shape[2], x.shape[3])

    b = np.array([[1, 2, 1], [0,0,0], [-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).to(gpu_id)
    conv2.weight = nn.Parameter(torch.from_numpy(b).type(torch.cuda.FloatTensor).unsqueeze(0).unsqueeze(0))
    G_y = conv2(Variable(x)).data.view(1, x.shape[2], x.shape[3])

    G = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    return G

class angular_loss(torch.nn.Module):
    
    def __init__(self):
        super(angular_loss,self).__init__()
        
    def forward(self, illum_gt, illum_pred):
        # img_gt = img_input / illum_gt
        # illum_gt = img_input / img_gt
        # illum_pred = img_input / img_output
	
        # ACOS
        cos_between = torch.nn.CosineSimilarity(dim=1)
        cos = cos_between(illum_gt, illum_pred)
        cos = torch.clamp(cos,-0.99999, 0.99999)
        loss = torch.mean(torch.acos(cos)) * 180 / pi

	# MSE
        # loss = torch.mean((illum_gt - illum_pred)**2)
	
        # 1 - COS
        # loss = 1 - torch.mean(cos)
	
        # 1 - COS^2
        # loss = 1 - torch.mean(cos**2)
        return loss
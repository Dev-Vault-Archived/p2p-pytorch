import torch
import torch.nn as nn
import numpy as np
from math import pi
from torch.nn import init
# import torch.nn.functional as F
from torch.autograd import Variable
import functools
# import random
from torch.optim import lr_scheduler

import torchvision.models as models
import torchvision.transforms as transforms

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

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


def define_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = None

    # norm_layer = get_norm_layer(norm_type='batch')
    # net = ResnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)
    net = TransformNetwork()

    return init_net(net, init_type, init_gain, gpu_id)

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# class ResnetGenerator(nn.Module):
#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
#         assert(n_blocks >= 0)
#         super(ResnetGenerator, self).__init__()
#         self.input_nc = input_nc
#         self.output_nc = output_nc
#         self.ngf = ngf
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         self.inc = Inconv(input_nc, ngf, norm_layer, use_bias)
#         self.down1 = Down(ngf, ngf * 2, norm_layer, use_bias)
#         self.down2 = Down(ngf * 2, ngf * 4, norm_layer, use_bias)

#         model = []
#         for i in range(n_blocks):
#             model += [ResBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
#         self.resblocks = nn.Sequential(*model)

#         self.up1 = Up(ngf * 4, ngf * 2, norm_layer, use_bias)
#         self.up2 = Up(ngf * 2, ngf, norm_layer, use_bias)

#         self.outc = Outconv(ngf, output_nc)

#     def forward(self, input):
#         out = {}
#         out['in'] = self.inc(input)
#         out['d1'] = self.down1(out['in'])
#         out['d2'] = self.down2(out['d1'])
#         out['bottle'] = self.resblocks(out['d2'])
#         out['u1'] = self.up1(out['bottle'])
#         out['u2'] = self.up2(out['u1'])

#         return self.outc(out['u2'])

# class Inconv(nn.Module):
#     def __init__(self, in_ch, out_ch, norm_layer, use_bias):
#         super(Inconv, self).__init__()
#         self.inconv = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0,
#                       bias=use_bias),
#             norm_layer(out_ch),
#             nn.ReLU(True)
#         )

#     def forward(self, x):
#         x = self.inconv(x)
#         return x


# class Down(nn.Module):
#     def __init__(self, in_ch, out_ch, norm_layer, use_bias):
#         super(Down, self).__init__()
#         self.down = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3,
#                       stride=2, padding=1, bias=use_bias),
#             norm_layer(out_ch),
#             nn.ReLU(True)
#         )

#     def forward(self, x):
#         x = self.down(x)
#         return x

# # Define a Resnet block
# class ResBlock(nn.Module):
#     def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         super(ResBlock, self).__init__()
#         self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

#     def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         conv_block = []
#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)

#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
#                        norm_layer(dim),
#                        nn.ReLU(True)]
#         if use_dropout:
#             conv_block += [nn.Dropout(0.5)]

#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
#                        norm_layer(dim)]

#         return nn.Sequential(*conv_block)

#     def forward(self, x):
#         out = x + self.conv_block(x)
#         return nn.ReLU(True)(out)


# class Up(nn.Module):
#     def __init__(self, in_ch, out_ch, norm_layer, use_bias):
#         super(Up, self).__init__()
#         self.up = nn.Sequential(
#             nn.ConvTranspose2d(in_ch, out_ch,
#                                kernel_size=3, stride=2,
#                                padding=1, output_padding=1,
#                                bias=use_bias),
#             norm_layer(out_ch),
#             nn.ReLU(True)
#         )

#     def forward(self, x):
#         x = self.up(x)
#         return x


# class Outconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(Outconv, self).__init__()
#         self.outconv = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.outconv(x)
#         return x

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

# Image Transform Network
class TransformNetwork(nn.Module):
    def __init__(self):
        super(TransformNetwork, self).__init__()
        
        # nonlineraity
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.25)
        self.tanh = nn.Tanh()

        # encoding layers
        self.conv1 = ConvLayer(3, 64, kernel_size=9, stride=1)
        self.in1_e = nn.BatchNorm2d(64, affine=True)

        self.conv2 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in2_e = nn.BatchNorm2d(128, affine=True)

        self.conv3 = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.in3_e = nn.BatchNorm2d(256, affine=True)

        # residual layers
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        self.res4 = ResidualBlock(256)
        self.res5 = ResidualBlock(256)
        self.res6 = ResidualBlock(256)
        self.res7 = ResidualBlock(256)
        self.res8 = ResidualBlock(256)
        self.res9 = ResidualBlock(256)

        # decoding layers
        self.deconv3 = UpsampleConvLayer(256, 128, kernel_size=3, stride=1, upsample=2 )
        self.in3_d = nn.BatchNorm2d(128, affine=True)

        self.deconv2 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2 )
        self.in2_d = nn.BatchNorm2d(64, affine=True)

        self.deconv1 = UpsampleConvLayer(64, 3, kernel_size=9, stride=1)
        self.in1_d = nn.BatchNorm2d(3, affine=True)

    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))

        # residual layers
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)
        y = self.res8(y)
        y = self.res9(y)

        # decode
        y = self.leakyRelu(self.in3_d(self.deconv3(y)))
        y = self.leakyRelu(self.in2_d(self.deconv2(y)))
        y = self.tanh(self.in1_d(self.deconv1(y)))
        # y = self.deconv1(y)

        return y

# class TransformNetwork(nn.Module):    
#     def __init__(self):        
#         super(TransformNetwork, self).__init__()        
        
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
    norm_layer = get_norm_layer(norm_type=norm)

    net = MultiscaleDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, num_D=3, getIntermFeat=True)

    return init_net(net, init_type, init_gain, gpu_id)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
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
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
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
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
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
    def __init__(self, use_lsgan=True, target_real_label=0.9, target_fake_label=0.0,
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
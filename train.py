from __future__ import print_function
import argparse
import os
from math import isnan, log10
from numpy.core.numeric import Inf

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from skimage.metrics import structural_similarity

# from utils import save_img
from PIL import Image
from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate, angular_loss, sobelLayer
from data import get_training_set, get_test_set

import warnings

warnings.filterwarnings("ignore")

mse_criterion = torch.nn.MSELoss(reduction='mean')

def tensor2img(tensor):
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    tensor = np.squeeze(tensor)
    tensor = np.moveaxis(tensor, 0, 2)
    tensor = tensor * 255
    tensor = tensor.clip(0, 255).astype(np.uint8)
    
    img = Image.fromarray(tensor)
    return img

def ssim(image_out, image_ref):
    image_out = np.array(tensor2img(image_out), dtype='float')
    image_ref = np.array(tensor2img(image_ref), dtype='float')

    return structural_similarity(image_out, image_ref, multichannel=True)

def psnr(ground, compressed):
    np_ground = np.array(tensor2img(ground), dtype='float')
    np_compressed = np.array(tensor2img(compressed), dtype='float')
    mse = np.mean((np_ground - np_compressed)**2)
    psnr = np.log10(255**2/mse) * 10
    return psnr

def extract_features(model, x, layers):
    features = list()

    # normalize image to match vgg network
    x = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x)

    for index, layer in enumerate(model):
        x = layer(x)
        if index in layers:
            features.append(x)
    return features

def gram(x):
    b ,c, h, w = x.size()
    g = torch.bmm(x.view(b, c, h*w), x.view(b, c, h*w).transpose(1,2))
    return g.div(h*w)

def calc_Gram_Loss(features, targets, weights=None):
    if weights is None:
        weights = [1/len(features)] * len(features)
        
    gram_loss = 0
    for f, t, w in zip(features, targets, weights):
        gram_loss += mse_criterion(gram(f), gram(t)) * w
    return gram_loss

def calc_c_loss(features, targets, weights=None):
    if weights is None:
        weights = [1/len(features)] * len(features)
    
    content_loss = 0
    for f, t, w in zip(features, targets, weights):
        content_loss += mse_criterion(f, t) * w
        
    return content_loss

def load_checkpoint(net_g, net_d, opt_g, opt_d, sched_g, sched_d, loss_logger, filename='net_epoch_x.pth'):
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> Loading checkpoint '{}'".format(filename))
        state = torch.load(filename)

        start_epoch = state['epoch']
        net_g.load_state_dict(state['state_dict_g'])
        net_d.load_state_dict(state['state_dict_d'])
        opt_g.load_state_dict(state['optimizer_g'])
        opt_d.load_state_dict(state['optimizer_d'])
        sched_g.load_state_dict(state['scheduler_g'])
        sched_d.load_state_dict(state['scheduler_d'])
        loss_logger = state['losslogger']
    else:
        print("=> No checkpoint found at '{}'".format(filename))
        exit()

    return start_epoch, net_g, net_d, opt_g, opt_d, sched_g, sched_d, loss_logger

def calc_tv_Loss(x):
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss

if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    # Training settings
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    parser.add_argument('--dataset', required=True, help='facades')
    parser.add_argument('--name', required=True, help='training name')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
    parser.add_argument('--nepoch', type=int, default=50, help='# of epoch')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--epochsave', type=int, default=50, help='test')

    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
    parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
    opt = parser.parse_args()

    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    cudnn.benchmark = True

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')
    root_path = "dataset/"
    train_set = get_training_set(root_path + opt.dataset, opt.direction)
    test_set = get_test_set(root_path + opt.dataset, opt.direction)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    print('===> Building models')

    sobelLambda = 0

    net_g = define_G('normal', 0.02, gpu_id=device)
    net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, gpu_id=device)

    criterionGAN = GANLoss().to(device)
    criterionFeat = nn.L1Loss().to(device)
    # criterionL1 = nn.L1Loss().to(device)
    # criterionMSE = nn.MSELoss().to(device)
    criterionAngular = angular_loss().to(device)

    criterionVGG = torchvision.models.vgg16(pretrained=True).features.to(device)
    # class FeatureExtractor(nn.Module):
    #     def __init__(self, cnn, feature_layer=11):
    #         super(FeatureExtractor, self).__init__()
    #         self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])

    #     def normalize(self, tensors, mean, std):
    #         if not torch.is_tensor(tensors):
    #             raise TypeError('tensor is not a torch image.')
    #         for tensor in tensors:
    #             for t, m, s in zip(tensor, mean, std):
    #                 t.sub_(m).div_(s)
    #         return tensors

    #     def forward(self, x):
    #         # it image is gray scale then make it to 3 channel
    #         if x.size()[1] == 1:
    #             x = x.expand(-1, 3, -1, -1)
                
    #         # [-1: 1] image to  [0:1] image---------------------------------------------------(1)
    #         x = (x + 1) * 0.5
            
    #         # https://pytorch.org/docs/stable/torchvision/models.html
    #         x.data = self.normalize(x.data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #         return self.features(x)

    # # Feature extracting using vgg19
    # vgg19 = torchvision.models.vgg19(pretrained=True)
    # feature_extractor = FeatureExtractor(vgg19, feature_layer=35).to(device)

    # class VGG19Loss(object):
    #     def __call__(self, output, target):
        
    #         # [-1: 1] image to  [0:1] image---------------------------------------------------(2)
    #         output = (output + 1) * 0.5
    #         target = (target + 1) * 0.5

    #         output = feature_extractor(output)
    #         target = feature_extractor(target).data
    #         return MSE(output, target)

    # # criterion
    # MSE = nn.MSELoss().to(device)
    # BCE = nn.BCELoss().to(device)
    # criterionVGG = VGG19Loss()

    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, opt)
    net_d_scheduler = get_scheduler(optimizer_d, opt)

    losslogger = []
    start_epoch = opt.epoch_count

    if start_epoch > 1:
        # Jika ternyata start epochnya lebih dari 1, berarti load checkpoint
        start_epoch, net_g, net_d, optimizer_g, optimizer_d, net_g_scheduler, net_d_scheduler, losslogger = load_checkpoint(net_g, net_d, optimizer_g, optimizer_d, net_g_scheduler, net_d_scheduler, losslogger, "checkpoint/{}/net_{}_epoch_{}.pth".format(opt.dataset, opt.name, start_epoch-1))

        for state in optimizer_g.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        for state in optimizer_d.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # num_epoch = opt.niter + opt.niter_decay + 1
    num_epoch = opt.nepoch + 1
    for epoch in range(start_epoch, num_epoch):
        # train
        bar = tqdm(enumerate(training_data_loader, 1))
        data_len = len(training_data_loader)

        # losses sum
        sum_d_loss = 0
        sum_g_loss = 0
        sum_tot_g_loss = 0

        sum_gfeat_loss = 0

        sum_angular_loss = 0
        # sum_sobel_loss = 0

        sum_perp_loss = 0
        sum_style_loss = 0
        sum_tv_loss = 0

        for iteration, batch in bar:

            # forward
            real_a, real_b = batch[0].to(device), batch[1].to(device)
            # Generate fake real image
            fake_b = net_g(real_a)

            # Updating Detection network (Discriminator)

            optimizer_d.zero_grad()
            
            # Train discriminator with with fake image and loss

            # sobel layer
            # fake_sobel = sobelLayer(fake_b, gpu_id=device)

            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab.detach())
            loss_d_fake = criterionGAN(pred_fake, False)

            # Train discriminator with real image and loss
            # real_sobel = sobelLayer(real_b, gpu_id=device).detach()

            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = net_d.forward(real_ab)
            loss_d_real = criterionGAN(pred_real, True)
            
            # Combined D loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            loss_d.backward()
        
            optimizer_d.step()

            # Update generation network

            optimizer_g.zero_grad()

            # Masking real_a and fake_b

            # First, G(A) should fake the discriminator
            masking = torch.bitwise_and(fake_b, real_a)
            # mask_image = transforms.ToTensor()(masking).unsqueeze_(0).to(device)

            # save_img(fake_b.detach().squeeze(0).cpu(), "fake_b.png")
            # save_img(real_a.detach().squeeze(0).cpu(), "real_a.png")
            # save_img(mask_image.detach().squeeze(0).cpu(), "mask.png")

            fake_ab = torch.cat((real_a, masking), 1)
            pred_fake = net_d.forward(fake_ab)
            loss_g_gan = criterionGAN(pred_fake, True)

            # Second, G(A) = B
            # loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb

            # GAN Feature matching loss
            loss_G_GAN_Feat = 0
            N_Layers_D = 3
            Num_D = 3
            feat_weights = 4.0 / (N_Layers_D + 1)
            D_weights = 1.0 / Num_D
            for i in range(Num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * 10.0
            
            loss_g = loss_g_gan + loss_G_GAN_Feat

            eps = torch.tensor(1e-04).to(device)
            illum_gt = torch.div(real_a, torch.max(real_b, eps))
            illum_pred = torch.div(real_a, torch.max(fake_b, eps))
            loss_G_Ang = criterionAngular(illum_gt, illum_pred) * 1.0

            loss_g += loss_G_Ang

            # loss_sobelL1 = criterionFeat(fake_sobel, real_sobel) * sobelLambda
            # loss_g += loss_sobelL1

            # Perceptual loss
            # perp_loss = criterionVGG(fake_b, real_b)

            # loss_g += perp_loss * 10

            target_content_features = extract_features(criterionVGG, real_b, [15])
            target_style_features = extract_features(criterionVGG, real_b, [3, 8, 15, 22]) 

            output_content_features = extract_features(criterionVGG, fake_b, [15])
            output_style_features = extract_features(criterionVGG, fake_b, [3, 8, 15, 22])

            style_loss = calc_Gram_Loss(output_style_features, target_style_features)
            content_loss = calc_c_loss(output_content_features, target_content_features)
            tv_loss = calc_tv_Loss(fake_b)

            # # loss_g += content_loss * 1.0 + tv_loss * 1.0
            loss_g += content_loss * 15.0 + style_loss * 15.0 + tv_loss * 1.0
            # loss_g += style_loss * 10.0

            loss_g.backward()

            optimizer_g.step()

            sum_d_loss += loss_d.item()
            sum_g_loss += loss_g_gan.item()
            sum_gfeat_loss += loss_G_GAN_Feat.item()
            sum_angular_loss += loss_G_Ang.item()
            # sum_sobel_loss += loss_sobelL1.item()
            sum_perp_loss += content_loss.item()
            sum_style_loss += style_loss.item()
            sum_tv_loss += tv_loss.item()

            sum_tot_g_loss += loss_g.item()

            # In last iteration
            if iteration == data_len:
                # Pass for now
                pass
            
            bar.set_description(desc='itr: %d/%d [%3d/%3d] [D: %.6f] [G: %.6f] [GF: %.6f] [A: %.6f] [C: %.6f] [S: %.6f] [TV: %.6f] [Tot: %.6f]' %(
                iteration,
                data_len,
                epoch,
                num_epoch - 1,
                sum_d_loss/max(1, iteration),
                sum_g_loss/max(1, iteration),
                sum_gfeat_loss/max(1, iteration),
                sum_angular_loss/max(1, iteration),
                # sum_sobel_loss/max(1, iteration),
                sum_perp_loss/max(1, iteration),
                sum_style_loss/max(1, iteration),
                sum_tv_loss/max(1, iteration),
                sum_tot_g_loss/max(1, iteration),
            ))

            # print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} Loss_GFeat: {:.4f} Loss_Sobel: {:.4f} Loss_Perp: {:.4f} Loss_TV: {:.4f}".format(
                # epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item(), loss_G_GAN_Feat.item(), loss_sobelL1.item(), content_loss.item(), tv_loss.item()))

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        # if epoch <= 20:
        #     sobelLambda = 100/20*epoch

        #     print('Update sobel lambda: %f' % (sobelLambda))

        # test
        psnr_list = []
        ssim_list = []
        max_psnr = 0
        max_ssim = 0
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = net_g(input)

            # save_img(input.detach().squeeze(0).cpu(), "in.png")
            # save_img(target.detach().squeeze(0).cpu(), "tar.png")
            # save_img(prediction.detach().squeeze(0).cpu(), "pred.png")

            # mse = criterionMSE(prediction, target)
            # psnr = 10 * log10(1 / mse.item())
            peesneen = psnr(target, prediction)
            esesim = ssim(prediction, target)

            if (peesneen >= Inf):
                # change the infinity to some number
                peesneen = 60.0

            max_psnr = max(max_psnr, peesneen)
            max_ssim = max(max_ssim, esesim)

            psnr_list.append(peesneen)
            ssim_list.append(esesim)

        print("===> Avg. PSNR: {:.4f} dB".format(np.mean(psnr_list)))
        print("===> Avg. SSIM: {:.4f}".format(np.mean(ssim_list)))

        print("===> Max PSNR: {:.4f} dB".format(max_psnr))
        print("===> Max SSIM: {:.4f}".format(max_ssim))

        #checkpoint
        if epoch % opt.epochsave == 0:
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
                os.mkdir(os.path.join("checkpoint", opt.dataset))
            net_g_model_out_path = "checkpoint/{}/net_{}_epoch_{}.pth".format(opt.dataset, opt.name, epoch)
            # net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)

            # This is the state
            state = {
                'epoch': epoch + 1,
                'state_dict_g': net_g.state_dict(),
                # 'state_dict_d': net_d.state_dict(),
                # 'optimizer_g': optimizer_g.state_dict(),
                # 'optimizer_d': optimizer_d.state_dict(),
                # 'scheduler_g': net_g_scheduler.state_dict(),
                # 'scheduler_d': net_d_scheduler.state_dict(),
                # 'losslogger': losslogger,
            }

            torch.save(state, net_g_model_out_path)
            # torch.save(net_d, net_d_model_out_path)
            print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))
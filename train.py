from __future__ import print_function
import argparse
import os
from math import log10

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision

# from utils import save_img
from PIL import Image
from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate, angular_loss
from data import get_training_set, get_test_set

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

def psnr(ground, compressed):
    np_ground = np.array(tensor2img(ground), dtype='float')
    np_compressed = np.array(tensor2img(compressed), dtype='float')
    mse = np.mean((np_ground - np_compressed)**2)
    psnr = np.log10(255**2/mse) * 10
    return psnr

def extract_features(model, x, layers):
    features = list()
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

def calc_tv_Loss(x):
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss

if __name__ == '__main__':

    # torch.autograd.set_detect_anomaly(True)

    # Training settings
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    parser.add_argument('--dataset', required=True, help='facades')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
    parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
    parser.add_argument('--epochsave', type=int, default=50, help='test')
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

    # sobelLambda = 0

    net_g = define_G('normal', 0.02, gpu_id=device)
    net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, gpu_id=device)

    criterionGAN = GANLoss().to(device)
    criterionFeat = nn.L1Loss().to(device)
    # criterionL1 = nn.L1Loss().to(device)
    # criterionMSE = nn.MSELoss().to(device)
    criterionAngular = angular_loss().to(device)

    criterionVGG = torchvision.models.vgg19(pretrained=True).features.to(device)

    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, opt)
    net_d_scheduler = get_scheduler(optimizer_d, opt)

    num_epoch = opt.niter + opt.niter_decay + 1
    for epoch in range(opt.epoch_count, num_epoch):
        # train
        bar = tqdm(enumerate(training_data_loader, 1))
        data_len = len(training_data_loader)

        # losses sum
        sum_d_loss = 0
        sum_g_loss = 0

        sum_gfeat_loss = 0

        sum_angular_loss = 0

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
            masking = np.bitwise_and(tensor2img(fake_b), tensor2img(real_a))
            mask_image = transforms.ToTensor()(masking).unsqueeze_(0).to(device)

            # save_img(fake_b.detach().squeeze(0).cpu(), "fake_b.png")
            # save_img(real_a.detach().squeeze(0).cpu(), "real_a.png")
            # save_img(mask_image.detach().squeeze(0).cpu(), "mask.png")

            fake_ab = torch.cat((real_a, mask_image), 1)
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
            loss_G_Ang = criterionAngular(illum_gt, illum_pred) * 1

            loss_g += loss_G_Ang

            # loss_sobelL1 = criterionFeat(fake_sobel, real_sobel) * sobelLambda
            # loss_g += loss_sobelL1

            # Perceptual loss
            target_content_features = extract_features(criterionVGG, real_b, [15])
            # target_style_features = extract_features(criterionVGG, real_b, [3, 8, 15, 22]) 

            output_content_features = extract_features(criterionVGG, fake_b, [15])
            # output_style_features = extract_features(criterionVGG, fake_b, [3, 8, 15, 22])

            # style_loss = calc_Gram_Loss(output_style_features, target_style_features)
            content_loss = calc_c_loss(output_content_features, target_content_features)
            tv_loss = calc_tv_Loss(fake_b)

            loss_g += content_loss * 1.0 + tv_loss * 1.0
            # loss_g += content_loss * 1.0 + tv_loss * 1.0 + style_loss * 10.0
            # loss_g += style_loss * 10.0

            loss_g.backward()

            optimizer_g.step()
            
            sum_d_loss += loss_d.item()
            sum_g_loss += loss_g.item()
            sum_gfeat_loss += loss_G_GAN_Feat.item()
            sum_angular_loss += loss_G_Ang.item()
            sum_perp_loss += content_loss.item()
            # sum_style_loss += style_loss.item()
            sum_tv_loss += tv_loss.item()

            bar.set_description(desc='itr: %d/%d [%3d/%3d] [D Loss: %.6f] [G Loss: %.6f] [GFeat Loss: %.6f] [Ang Loss: %.6f] [Perp Loss: %.6f] [Style Loss: %.6f] [TV Loss: %.6f]' %(
                iteration,
                data_len,
                epoch,
                num_epoch,
                sum_d_loss/max(1, iteration),
                sum_g_loss/max(1, iteration),
                sum_g_loss/max(1, iteration),
                sum_angular_loss/max(1, iteration),
                sum_perp_loss/max(1, iteration),
                sum_style_loss/max(1, iteration),
                sum_tv_loss/max(1, iteration)
            ))
            # print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} Loss_GFeat: {:.4f} Loss_Sobel: {:.4f} Loss_Perp: {:.4f} Loss_TV: {:.4f}".format(
                # epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item(), loss_G_GAN_Feat.item(), loss_sobelL1.item(), content_loss.item(), tv_loss.item()))

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        if epoch <= 20:
            sobelLambda = 100/20*epoch

            print('Update sobel lambda: %f' % (sobelLambda))

        # test
        avg_psnr = 0
        max_psnr = 0
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = net_g(input)

            # save_img(input.detach().squeeze(0).cpu(), "in.png")
            # save_img(target.detach().squeeze(0).cpu(), "tar.png")
            # save_img(prediction.detach().squeeze(0).cpu(), "pred.png")

            # mse = criterionMSE(prediction, target)
            # psnr = 10 * log10(1 / mse.item())
            peesneen = psnr(target, prediction)

            max_psnr = max(max_psnr, peesneen)

            avg_psnr += peesneen
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
        print("===> Max PSNR: {:.4f} dB".format(max_psnr))

        #checkpoint
        if epoch % opt.epochsave == 0:
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
                os.mkdir(os.path.join("checkpoint", opt.dataset))
            net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
            net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
            torch.save(net_g.state_dict(), net_g_model_out_path)
            # torch.save(net_d, net_d_model_out_path)
            print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))
"""
Author: Dr. Jin Zhang
E-mail: j.zhang@kust.edu.cn
Dept: Kunming University of Science and Technology
Codes for "Data-Driven Industrial Process Monitoring Using Deep Learning Features and Handcrafted Features"
Created on 2023.05.04
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
from torchvision.utils import save_image

import os
import time
import numpy
import argparse
from tqdm import tqdm

from dataset import Data4BiAAE
from models import *
from ganlosses import *
from util import AverageMeter

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/media/neuralits/Data_SSD/FrothData/Data4FrothGrade',
                        help='data')
    parser.add_argument('--csvfile', type=str,
                        default='/media/neuralits/Data_SSD/FrothData/XRFImgData4FrothStatusModel.csv', help='data')
    parser.add_argument('--batch-size', type=int, default=5, help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=600, help='number of epochs to train (default: 10)')
    parser.add_argument('--load-epoch', type=int, default=1, help='number of epochs to train (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='number of epochs to train (default: 10)')
    parser.add_argument('--learning-rate-Dz', type=float, default=1.5e-6,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--learning-rate-Dx', type=float, default=1e-4, help='number of epochs to train (default: 10)')
    parser.add_argument('--num-workers', type=int, default=4, help='number of epochs to train (default: 10)')
    parser.add_argument('--label-smoothing', type=int, default=0.9, help='One-sided label smoothing')
    parser.add_argument('--img_height', type=int, default=256, help='size of image height')
    parser.add_argument('--img_width', type=int, default=256, help='size of image width')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='interval between model checkpoints')
    parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
    parser.add_argument('--name', type=str, default='SS-BiAAE', help='It decides where to store samples and models')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    args = parser.parse_args()

    return args


def set_loader(args):
    data = Data4BiAAE()
    dataloader = DataLoader(data, args.batch_size, shuffle=False, num_workers=args.num_workers)
    return dataloader


def set_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder()
    decoder = Generator()
    dis_latent_model = Discriminator4z()
    dis_img_model = Discriminator4x()
    feats_model = FeatureExtractor()

    if args.load_epoch != 1:
        encoder.load_state_dict(torch.load('./saved_models/encoder_%d.pth' % (args.load_epoch)))  # 2150
        decoder.load_state_dict(torch.load('./saved_models/decoder_%d.pth' % (args.load_epoch)))  # 2150
        # dis_img_model.load_state_dict(torch.load('./saved_models/discriminator4x1_%d.pth' % (2150) )) #2150
        dis_img_model.load_state_dict(torch.load('./saved_models/dis_img_model_%d.pth' % (args.load_epoch)))  # 2150
        dis_latent_model.load_state_dict(torch.load('./saved_models/dis_latent_model_%d.pth' % (args.load_epoch)))

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    dis_latent_model = dis_latent_model.to(device)
    dis_img_model = dis_img_model.to(device)
    feats_model = feats_model.to(device)

    return encoder, decoder, dis_latent_model, dis_img_model, feats_model


def set_optimizer(args, encoder, decoder, dis_latent_model, dis_img_model, feats_model):
    im_advers_loss_func = RelativisticAverageHingeGAN(dis_img_model)
    laten_advers_loss_func = LSGAN(dis_latent_model)
    percep_loss_func = Perceptual(feats_model)
    recon_loss_func = PixelwiseL1()
    self_sup_loss_func = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    optimizer_latent_dis = torch.optim.Adam(dis_latent_model.parameters(), lr=args.learning_rate / 10, weight_decay=1e-5)
    optimizer_im_dis = torch.optim.Adam(dis_img_model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    return im_advers_loss_func, laten_advers_loss_func, percep_loss_func, recon_loss_func, self_sup_loss_func \
        , optimizer_enc, optimizer_dec, optimizer_latent_dis, optimizer_im_dis


def denormalize4img(x_hat):
    mean = [0.5561, 0.5706, 0.5491]
    std = [0.1833, 0.1916, 0.2061]
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
    x = x_hat * std + mean
    return x


def train(args, writer, denormalize4img, dataloader, encoder, decoder, dis_latent_model, dis_img_model, im_advers_loss_func, \
            laten_advers_loss_func, percep_loss_func, recon_loss_func, self_sup_loss_func, optimizer_enc, \
            optimizer_dec, optimizer_latent_dis, optimizer_im_dis):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for ii, img in tqdm(enumerate(dataloader)):
        iter_start_time = time.time()
        img = img.to(device)
        B, F, C, H, W = img.size()
        img = img.reshape(B * F, C, H, W)

        images = [img] + [nn.functional.avg_pool2d(img, int(np.power(2, i))) for i in range(1, 7)]
        real_images = list(reversed(images))

        feature = encoder(img)
        fake_images = decoder(feature)

        bf, c, h, w = feature.size()
        feature = feature.reshape(B, F, c, h, w)
        anchor_samp = feature[:, 0, :, :, :]
        posit_samp = feature[:, 1, :, :, :]
        negti_samp = feature[:, 2, :, :, :]

        syn_latents = 0.6 * anchor_samp.detach() + 0.4 * negti_samp.detach()
        syn_images = decoder(syn_latents)

        feature = feature.reshape(bf, c, h, w)
        latent = torch.FloatTensor(numpy.random.normal(0, 1, (feature.size()))).to(device)

        # Updating encoder and decoder
        recon_loss = 4 * recon_loss_func.gen_loss(real_images, fake_images)  # cycle-consistency
        laten_recon_loss = recon_loss_func.gen_loss(syn_latents, encoder(syn_images[6]))
        percep_loss = 10 * percep_loss_func.gen_loss(img, fake_images[6])
        im_advers_loss = 0.5 * im_advers_loss_func.gen_loss(real_images, fake_images)
        laten_advers_loss = laten_advers_loss_func.gen_loss(latent, feature)
        self_sup_loss = self_sup_loss_func(anchor_samp.view(B, -1), posit_samp.view(B, -1), negti_samp.view(B, -1))
        latent_im_advers_loss = 0.5 * im_advers_loss_func.gen_loss(real_images, syn_images)

        losses = recon_loss + laten_recon_loss + percep_loss + im_advers_loss + laten_advers_loss + self_sup_loss + latent_im_advers_loss

        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        for p in dis_img_model.parameters():
            p.requires_grad = False
        for p in dis_latent_model.parameters():
            p.requires_grad = False
        losses.backward(retain_graph=True)
        optimizer_enc.step()
        optimizer_dec.step()

        # Updating dis_img_model
        fake_images = decoder(feature)
        fake_images = list(map(lambda x: x.detach(), fake_images))
        for p in dis_img_model.parameters():
            p.requires_grad = True
        optimizer_im_dis.zero_grad()
        dis_im_advers_loss = im_advers_loss_func.dis_loss(real_images, fake_images)
        dis_im_advers_loss.backward()
        optimizer_im_dis.step()

        # Updating dis_latent_model
        for p in dis_latent_model.parameters():
            p.requires_grad = True
        optimizer_latent_dis.zero_grad()
        dis_laten_advers_loss = 0.5 * laten_advers_loss_func.dis_loss(latent, feature)  # 保证real在前
        dis_laten_advers_loss.backward()
        optimizer_latent_dis.step()

        recon_losses.update(recon_loss.cpu().data, img.size(0))
        laten_recon_losses.update(laten_recon_loss.cpu().data, img.size(0))
        percep_losses.update(percep_loss.cpu().data, img.size(0))
        im_advers_losses.update(im_advers_loss.cpu().data, img.size(0))
        laten_advers_losses.update(laten_advers_loss.cpu().data, img.size(0))
        self_sup_losses.update(self_sup_loss.cpu().data, img.size(0))
        latent_im_advers_losses.update(latent_im_advers_loss.cpu().data, img.size(0))
        dis_im_advers_losses.update(dis_im_advers_loss.cpu().data, img.size(0))
        dis_laten_advers_losses.update(dis_laten_advers_loss.cpu().data, img.size(0))

        # for name, weight in decoder.named_parameters():
        #    if weight.requires_grad:
        #        print("weight.grad:", weight.grad.mean(), weight.grad.min(), weight.grad.max())

        for _, img in enumerate(dataloader):
            with torch.no_grad():
                img = img.to(device)
                B, F, C, H, W = img.size()
                img = img.reshape(B * F, C, H, W)
                feature = encoder(img)
                fake_images = decoder(feature)
                fake_images = list(map(lambda x: x.detach(), fake_images))
            break

        grid = torchvision.utils.make_grid(denormalize4img(fake_images[6].cpu().data))
        writer.add_image("decoded", grid, epoch)
        grid = torchvision.utils.make_grid(denormalize4img(syn_images[6].cpu().data))
        writer.add_image("synthesis", grid, epoch)
        grid = torchvision.utils.make_grid(denormalize4img(img.cpu().data))
        writer.add_image("real", grid, epoch)

        writer.add_scalar("recon_loss", recon_losses.avg, epoch)
        writer.add_scalar("laten_recon_loss", laten_recon_losses.avg, epoch)
        writer.add_scalar("percep_loss", percep_losses.avg, epoch)
        writer.add_scalar("im_advers_loss", im_advers_losses.avg, epoch)
        writer.add_scalar("laten_advers_loss", laten_advers_losses.avg, epoch)
        writer.add_scalar("self_sup_loss", self_sup_losses.avg, epoch)
        writer.add_scalar("latent_im_advers_loss", latent_im_advers_losses.avg, epoch)
        writer.add_scalar("dis_im_advers_loss", dis_im_advers_losses.avg, epoch)
        writer.add_scalar("dis_laten_advers_loss", dis_laten_advers_losses.avg, epoch)


def main():
    writer = SummaryWriter(comment="Mix-BiAAE")
    args = parse_option()
    dataloader = set_loader()
    encoder, decoder, dis_latent_model, dis_img_model, feats_model = set_model(args)
    im_advers_loss_func, laten_advers_loss_func, percep_loss_func, recon_loss_func, self_sup_loss_func \
        , optimizer_enc, optimizer_dec, optimizer_latent_dis, optimizer_im_dis = set_optimizer(args, encoder, decoder, dis_latent_model, dis_img_model, feats_model)

    iter_data_time = time.time()
    recon_losses = AverageMeter()
    percep_losses = AverageMeter()
    laten_recon_losses = AverageMeter()
    im_advers_losses = AverageMeter()
    laten_advers_losses = AverageMeter()
    self_sup_losses = AverageMeter()
    latent_im_advers_losses = AverageMeter()
    dis_im_advers_losses = AverageMeter()
    dis_laten_advers_losses = AverageMeter()

    for epoch in range(args.load_epoch, args.epochs + 1):
        img, fake_images, syn_images = train(args, writer, denormalize4img, dataloader, encoder, decoder, dis_latent_model, \
              dis_img_model, im_advers_loss_func, laten_advers_loss_func, percep_loss_func, recon_loss_func, self_sup_loss_func, \
              optimizer_enc, optimizer_dec, optimizer_latent_dis, optimizer_im_dis)

        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0 and epoch > args.load_epoch:
            torch.save(encoder.state_dict(), './saved_models/encoder_%d.pth' % (epoch))
            torch.save(decoder.state_dict(), './saved_models/decoder_%d.pth' % (epoch))
            torch.save(dis_latent_model.state_dict(), './saved_models/dis_latent_model_%d.pth' % (epoch))
            torch.save(dis_img_model.state_dict(), './saved_models/dis_img_model_%d.pth' % (epoch))

            pic = img.cpu().data
            save_image(pic, './dc_img/{}/{}_real_image.png'.format(args.name, epoch), nrow=6, normalize=True)
            pic = fake_images[6].cpu().data
            save_image(pic, './dc_img/{}/{}_fake_image.png'.format(args.name, epoch), nrow=6, normalize=True)
            pic = syn_images[6].cpu().data
            save_image(pic, './dc_img/{}/{}_syn_image.png'.format(args.name, epoch), nrow=6, normalize=True)
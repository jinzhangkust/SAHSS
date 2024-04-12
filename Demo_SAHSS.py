"""
Author: Dr. Jin Zhang 
E-mail: j.zhang@kust.edu.cn
Created on 2024.02.09
"""

import torch
import torchvision
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import os
import sys
import time
import argparse
import numpy as np

from dataset import TailingSensorSet
#from embedding_networks import TransformerEmbedding
from momentum_memory_network import Memory_Network
from util import AverageMeter
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_percentage_error

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=150, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers=4*num_GPU')
    parser.add_argument('--epoch', type=int, default=1, help='number of training epochs')
    parser.add_argument('--total_epochs', type=int, default=300, help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # model dataset
    parser.add_argument('--model_name', type=str, default='SAHSS')
    # checkpoint
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    # memory
    parser.add_argument('--mem_size', type=int, default=1500, help='number of memory slots')  # 2000
    parser.add_argument('--key_dim', type=int, default=128, help='key dimension')
    parser.add_argument('--val_dim', type=int, default=3, help='dimension of class distribution')
    parser.add_argument('--top_k_r', type=int, default=32, help='top_k for memory reading')  # 200
    parser.add_argument('--val_thres', type=float, default=0.08, help='threshold for value matching')  # 0.06
    parser.add_argument('--val_ratio', type=float, default=0.4, help='threshold for value matching')  # 0.06
    parser.add_argument('--val_clip', type=float, default=0.4, help='threshold for value matching')
    parser.add_argument('--temp', type=float, default=0.015, help='temperature for softmax')
    parser.add_argument('--age_noise', type=float, default=8.0, help='number of training epochs')


    opt = parser.parse_args()
    
    opt.save_folder = os.path.join('./save', opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt
    
    
def set_loader(opt):
    full_data = TailingSensorSet(train_mode="train", clip_mode='single')
    train_size = int(0.6 * len(full_data))
    val_size = int(0.2 * len(full_data))
    test_size = len(full_data) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size],
                                                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return train_loader, val_loader, test_loader


def set_model(opt):
    model = Memory_Network(mem_size=opt.mem_size, key_dim=opt.key_dim, val_dim=opt.val_dim, top_k_r=opt.top_k_r,
                           temp=opt.temp, age_noise=opt.age_noise)

    criterion = torch.nn.MSELoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def set_optimizer(opt, model):
    optimizer = optim.Adam([{'params': model.encoder_q.projector.parameters(), 'lr': opt.learning_rate/20},
                            {'params': model.encoder_q.encoder.parameters(), 'lr': opt.learning_rate}],
                           weight_decay=opt.weight_decay)
    return optimizer


def cal_accuracy(out_hat, truth_hat):
    R2_0 = r2_score(out_hat[:, 0], truth_hat[:, 0])
    R2_1 = r2_score(out_hat[:, 1], truth_hat[:, 1])
    R2_2 = r2_score(out_hat[:, 2], truth_hat[:, 2])
    return R2_0, R2_1, R2_2


def cal_metrics(y_pred, y_true):
    # r2_score: R2
    R2_0 = r2_score(y_true[:, 0], y_pred[:, 0])
    R2_1 = r2_score(y_true[:, 1], y_pred[:, 1])
    R2_2 = r2_score(y_true[:, 2], y_pred[:, 2])
    # mean_squared_error (If True returns MSE value, if False returns RMSE value.): RMSE
    RMSE_0 = mean_squared_error(y_true[:, 0], y_pred[:, 0], squared=False)
    RMSE_1 = mean_squared_error(y_true[:, 1], y_pred[:, 1], squared=False)
    RMSE_2 = mean_squared_error(y_true[:, 2], y_pred[:, 2], squared=False)
    # mean_absolute_error: MAE
    MAE_0 = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    MAE_1 = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    MAE_2 = mean_absolute_error(y_true[:, 2], y_pred[:, 2])
    # explained_variance_score: EVS
    EVS_0 = explained_variance_score(y_true[:, 0], y_pred[:, 0])
    EVS_1 = explained_variance_score(y_true[:, 1], y_pred[:, 1])
    EVS_2 = explained_variance_score(y_true[:, 2], y_pred[:, 2])
    # mean_absolute_percentage_error: MAPE
    MAPE_0 = mean_absolute_percentage_error(y_true[:, 0], y_pred[:, 0])
    MAPE_1 = mean_absolute_percentage_error(y_true[:, 1], y_pred[:, 1])
    MAPE_2 = mean_absolute_percentage_error(y_true[:, 2], y_pred[:, 2])
    return R2_0, R2_1, R2_2, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2, EVS_0, EVS_1, EVS_2, MAPE_0, MAPE_1, MAPE_2


def train(train_loader, model, criterion, optimizer, epoch, opt, tb):
    model.train()

    val_thres_list = list()

    batch_time = AverageMeter()
    mse_losses = AverageMeter()
    entropy_losses = AverageMeter()
    contrast_losses = AverageMeter()
    mse_total_loss = 0
    entropy_total_loss = 0
    contrast_total_loss = 0
    # predict_set = []
    # target_set = []

    warm_weight = 0.2 if epoch > 50 else 0
    end = time.time()
    for idx, (reagents, images, hf_value, targets) in enumerate(train_loader):
        reagents = reagents.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        bsz = targets.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        query, key = model(reagents.float(), images)
        pred, softmax_score, softmax_score_mean = model.predict(query)
        mse_loss = criterion(pred, targets.float())
        entropy_loss = (-softmax_score * torch.log(softmax_score)).sum(dim=1).mean()
        contrast_loss = model.contrast_loss(query, targets, opt.val_thres)
        loss = mse_loss + 0*entropy_loss + warm_weight * contrast_loss

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # memory update
        with torch.no_grad():
            model.key_enc_update()
            query, key = model(reagents.float(), images)
            val_thres_temp = model.memory_update(query, key, targets, opt.val_thres, opt.val_ratio, opt.val_clip)
            val_thres_list.append(val_thres_temp.cpu().numpy())


        # update metric
        #print(f"softmax_score_mean: {softmax_score_mean.shape}")
        if idx:
            predict_set = np.append(predict_set, pred.detach().cpu().numpy(), axis=0)
            target_set = np.append(target_set, targets.cpu().numpy(), axis=0)
            softmax_score_mean_set = np.append(softmax_score_mean_set, softmax_score_mean.view(1,-1).detach().cpu().numpy(), axis=0)
        else:
            predict_set = pred.detach().cpu().numpy()
            target_set = targets.cpu().numpy()
            softmax_score_mean_set = softmax_score_mean.view(1,-1).detach().cpu().numpy()

        mse_losses.update(mse_loss.item(), bsz)
        entropy_losses.update(entropy_loss.item(), bsz)
        contrast_losses.update(contrast_loss.item(), bsz)

        mse_total_loss += mse_loss.item()
        entropy_total_loss += entropy_loss.item()
        contrast_total_loss += contrast_loss.item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # grid = torchvision.utils.make_grid(images)
    # tb.add_image("images", grid)
    # tb.add_graph(model, (tailings.float(),images))
    acc0, acc1, acc2 = cal_accuracy(predict_set, target_set)
    tb.add_scalar("Acc0", acc0, epoch)
    tb.add_scalar("Acc1", acc1, epoch)
    tb.add_scalar("Acc2", acc2, epoch)
    tb.add_scalar("MSE-Loss", mse_total_loss, epoch)
    tb.add_scalar("Entropy-Loss", entropy_total_loss, epoch)
    tb.add_scalar("Contrast-Loss", contrast_total_loss, epoch)

    val_thres_epoch_mean = np.mean(val_thres_list)
    val_thres_epoch_max = np.max(val_thres_list)
    return val_thres_epoch_mean, val_thres_epoch_max


def validate(val_loader, model, criterion, epoch, opt, tb):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    mse_losses = AverageMeter()
    contrast_losses = AverageMeter()

    mse_total_loss = 0
    contrast_total_loss = 0

    with torch.no_grad():
        end = time.time()
        for idx, (reagents, images, hf_value, targets) in enumerate(val_loader):
            reagents = reagents.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            bsz = targets.shape[0]

            # forward
            query, key = model(reagents.float(), images)
            pred,_,_ = model.predict(query)
            mse_loss = criterion(pred, targets.float())
            contrast_loss = model.contrast_loss(query, targets, opt.val_ratio)

            # update metric
            if idx:
                predict_set = np.append(predict_set, pred.detach().cpu().numpy(), axis=0)
                target_set = np.append(target_set, targets.cpu().numpy(), axis=0)
            else:
                predict_set = pred.detach().cpu().numpy()
                target_set = targets.cpu().numpy()

            mse_losses.update(mse_loss.item(), bsz)
            contrast_losses.update(contrast_loss.item(), bsz)

            mse_total_loss += mse_loss.item()
            contrast_total_loss += contrast_loss.item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    acc0, acc1, acc2 = cal_accuracy(predict_set, target_set)
    tb.add_scalar("Test-Acc0", acc0, epoch)
    tb.add_scalar("Test-Acc1", acc1, epoch)
    tb.add_scalar("Test-Acc2", acc2, epoch)
    tb.add_scalar("Test-MSE-Loss", mse_total_loss, epoch)
    tb.add_scalar("Test-Contrast-Loss", contrast_total_loss, epoch)
    # return losses.avg


def tenfoldtest(test_loader, model, tb):
    """validation"""
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (reagents, images, hf_value, targets) in enumerate(test_loader):
            reagents = reagents.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            bsz = targets.shape[0]

            # forward
            query, key = model(reagents.float(), images)
            output,_,_ = model.predict(query)

            if idx:
                predict_set = np.append(predict_set, output.detach().cpu().numpy(), axis=0)
                target_set = np.append(target_set, targets.cpu().numpy(), axis=0)
            else:
                predict_set = output.detach().cpu().numpy()
                target_set = targets.cpu().numpy()

    R2_0, R2_1, R2_2, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2, EVS_0, EVS_1, EVS_2, MAPE_0, MAPE_1, MAPE_2 = cal_metrics(
        predict_set, target_set)
    Bias_0 = predict_set[:, 0] - target_set[:, 0]
    Bias_1 = predict_set[:, 1] - target_set[:, 1]
    Bias_2 = predict_set[:, 2] - target_set[:, 2]

    for i in range(len(Bias_0)):
        tb.add_scalar("Test-Biase0", Bias_0[i], i)
        tb.add_scalar("Test-Biase1", Bias_1[i], i)
        tb.add_scalar("Test-Biase2", Bias_2[i], i)

    return R2_0, R2_1, R2_2, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2, EVS_0, EVS_1, EVS_2, MAPE_0, MAPE_1, MAPE_2


def cosine_rampdown(epoch, ranmpdown_length = 60):
    state = epoch % ranmpdown_length
    return 0.03 * float((np.cos(np.pi * state / ranmpdown_length) + 1) - 1.0)

def main():
    best_acc = 0
    opt = parse_option()
    tb = SummaryWriter(comment="SAHSS")

    train_loader, val_loader, test_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)

    val_thres_epoch_mean_last = 1.0
    for epoch in range(opt.epoch + 1, opt.total_epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)
        val_thres_epoch_mean, val_thres_epoch_max = train(train_loader, model, criterion, optimizer, epoch, opt, tb)
        opt.val_thres = min(val_thres_epoch_mean, val_thres_epoch_mean_last)
        opt.val_clip = (val_thres_epoch_max - val_thres_epoch_mean)*0.5 + val_thres_epoch_mean
        opt.val_clip = torch.from_numpy(np.array(opt.val_clip)).cuda()
        val_thres_epoch_mean_last = val_thres_epoch_mean
        opt.val_thres = opt.val_thres + cosine_rampdown(epoch)
        opt.val_thres = torch.clip(torch.from_numpy(np.array(opt.val_thres)).cuda(), 0.006, opt.val_clip)
        print(f"epoch: {epoch}   val_thres: {opt.val_thres}   val_thres_epoch_mean: {val_thres_epoch_mean}   val_clip: {opt.val_clip}")

        validate(val_loader, model, criterion, epoch, opt, tb)

        if epoch % opt.save_freq == 0 & epoch != opt.epoch:
            save_file = os.path.join(opt.save_folder, 'MoMeMo_q_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(model.encoder_q.state_dict(), save_file)
            save_file = os.path.join(opt.save_folder, 'MoMeMo_k_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(model.encoder_k.state_dict(), save_file)
            key_slots_np = model.key_slots.cpu().numpy()
            val_slots_np = model.val_slots.cpu().numpy()
            age_np = model.age.cpu().numpy()
            np.save(os.path.join(opt.save_folder, "MoMeMo_key_{epoch}.npy".format(epoch=epoch)), key_slots_np)
            np.save(os.path.join(opt.save_folder, "MoMeMo_val_{epoch}.npy".format(epoch=epoch)), val_slots_np)
            np.save(os.path.join(opt.save_folder, "MoMeMo_age_{epoch}.npy".format(epoch=epoch)), age_np)

        if epoch >= opt.total_epochs - 10:
            acc0, acc1, acc2, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2, EVS_0, EVS_1, EVS_2, MAPE_0, MAPE_1, MAPE_2 = tenfoldtest(
                test_loader, model, tb)
            print('epoch {}, acc0 {}, acc1 {}, acc2 {}'.format(epoch, acc0, acc1, acc2))
            print('epoch {}, MAE0 {}, MAE1 {}, MAE2 {}'.format(epoch, MAE_0, MAE_1, MAE_2))
            print('epoch {}, RMSE0 {}, RMSE1 {}, RMSE2 {}'.format(epoch, RMSE_0, RMSE_1, RMSE_2))


if __name__ == '__main__':
    main()

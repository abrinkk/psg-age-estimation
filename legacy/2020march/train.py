import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import ax
# from torch_lr_finder import LRFinder

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
import argparse

from config import Config
from psg_dataset import PSG_Dataset, PSG_epoch_Dataset, PSG_pretrain_Dataset, PSG_feature_Dataset
from m_psg2label import M_PSG2FEAT, M_FEAT2LABEL
from utils import HuberLoss, accuracy, analyze_num_workers, save_model_F, save_model_L, debug_model, plot_regression_pred, get_age_label_distribution, save_train_features, plot_train_epoch, plot_huber_loss

# Parser
parser = argparse.ArgumentParser(description='Train model based on setup in config.')
parser.add_argument('--pre_train', type=bool, default=True,
                    help='To pretrain model.')
parser.add_argument('--load_best_pre_train', type=bool, default=False,
                    help='Load the pretrained model that was best on validation set.')
parser.add_argument('--pre_train_continue', type=bool, default=False,
                    help='To continue preivously trained model')
parser.add_argument('--test_pre', type=bool, default=False,
                    help='To continue preivously trained model')
parser.add_argument('--save_feat', type=bool, default=False,
                    help='Overwrite model F features')
parser.add_argument('--train', type=bool, default=False,
                    help='To posttrain model.')
parser.add_argument('--bo', type=bool, default=False,
                    help='To perform bayesian hyperparameter optimization.')
parser.add_argument('--load_best_train', type=bool, default=False,
                    help='Load the trained model that was best on validation set.')
parser.add_argument('--train_continue', type=bool, default=False,
                    help='To continue preivously trained model')
parser.add_argument('--test', type=bool, default=False,
                    help='To continue preivously trained model')

args = parser.parse_args()


class multi_label_loss(nn.Module):
    '''
    loss over (age, bmi, sex)
    '''
    def __init__(self, device):
        super(multi_label_loss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss()
        self.L1Loss = nn.L1Loss()
        self.device = device
        # Correction of loss size (if predicting mean)
        self.multi_weights = [68.13, 20.13, 0.62]
        self.lr_finder = False

    def forward(self, y, t):

        # L1 age and bmi
        loss_age_l1 = self.L1Loss(y[:, 0], t[:, 0]) # Baseline 15.96
        loss_bmi_l1 = self.L1Loss(y[:, 1], t[:, 1]) # Baseline 5.97

        # Multi Huber-CE-loss
        loss_age = HuberLoss(y[:, 0], t[:, 0], 5, self.multi_weights[0], self.device)
        loss_bmi = HuberLoss(y[:, 1], t[:, 1], 5, self.multi_weights[1], self.device)
        loss_sex = self.CELoss(y[:, 2:], t[:, 2].long()).to(self.device) # Baseline 0.62
        loss_multi = 1/3 * (loss_age + loss_bmi + loss_sex / self.multi_weights[2])
        if self.lr_finder:
            return loss_multi
        return loss_multi, loss_age_l1, loss_bmi_l1, loss_sex

def pretrain(args, config, model_F, device, writer):
    
    # Parameters
    test_params = {'batch_size': 1,
                   'shuffle': True,
                   'num_workers': config.pre_n_workers}
    pretrain_params = {'batch_size': config.pre_batch_size,
                       'shuffle': True,
                       'num_workers': config.pre_n_workers}
    test_epoch_params = {'batch_size': config.pre_batch_size,
                       'shuffle': False,
                       'num_workers': config.pre_n_workers}
    
    # Generators
    train_set = PSG_pretrain_Dataset(config, 'train')
    train_gen = DataLoader(train_set, **pretrain_params)
    val_set = PSG_Dataset(config, 'val')
    val_gen = DataLoader(val_set, **test_params)
    
    # Optimization
    # TODO: Implement weight decay only for weight parameters
    optimizer_F = torch.optim.Adam(
    [{'params': [p for name, p in model_F.named_parameters() if 'weight' in name], 'weight_decay': config.l2},
    {'params': [p for name, p in model_F.named_parameters() if 'weight' not in name], 'weight_decay':0}],
    lr=config.pre_lr)
    loss_fn_F = multi_label_loss(device)
    L1Loss = nn.L1Loss()
    CELoss = nn.CrossEntropyLoss()
    
    # Continue previous pretraining
    if not args.pre_train:
        if args.load_best_pre_train:
            checkpoint = torch.load(config.model_F_path + '.tar')
            model_F = checkpoint['best_model']
            model_F.return_att_weights = False # debug
        return model_F
    elif args.pre_train_continue:
        checkpoint = torch.load(config.model_F_path + '.tar')
        start_epoch = checkpoint['epoch']
        model_F.load_state_dict(checkpoint['model_state_dict'])
        optimizer_F.load_state_dict(checkpoint['optimizer_state_dict'])
        train_iter = start_epoch * len(train_set) // config.pre_batch_size
        best_loss = checkpoint['best_loss']
        
    else:
        train_iter = 0
        checkpoint = 0
        start_epoch = 0
    
    # Learning rate finder
    # model_F.lr_finder = True
    # train_gen.dataset.lr_finder = True
    # loss_fn_F.lr_finder = True
    # lr_finder = LRFinder(model_F, optimizer_F, loss_fn_F, device=device)
    # lr_finder.range_test(train_gen, start_lr=1e-5, end_lr=1e2, num_iter=100)
    # lr_finder.plot() # to inspect the loss-learning rate graph
    # lr_finder.reset() # to reset the model and optimizer to their initial state
    # model_F.lr_finder = False
    # train_gen.dataset.lr_finder = False
    # loss_fn_F.lr_finder = False

    # Loop over epochs
    for epoch in range(start_epoch, config.pre_max_epochs):
        # Training
        train_loss = []
        train_age_l1_loss = []
        train_bmi_l1_loss = []
        train_sex_ce_loss = []
        train_sex_acc = []
        model_F.train()
        for psg, lab, _ in train_gen:
            # Zero Gradients
            optimizer_F.zero_grad()
            # Make predictions
            psg, lab = psg.to(device), lab.to(device)
            out_F, _ = model_F(psg)
            # Optimize network
            loss_F, loss_age_l1, loss_bmi_l1, loss_sex_ce = loss_fn_F(out_F, lab)
            sex_acc = accuracy(out_F[:, 2:],lab[:, 2])
            loss_F.backward()
            optimizer_F.step()
            # Track the loss
            train_loss.append(loss_F.item())
            train_iter += 1
            train_loss.append(loss_F.item())
            writer.add_scalar('PreTraining/MultiLoss',
                              np.array(loss_F.item()),
                              train_iter)

            train_age_l1_loss.append(loss_age_l1.item())
            writer.add_scalar('PreTraining/AgeL1Loss',
                              np.array(loss_age_l1.item()),
                              train_iter)

            train_bmi_l1_loss.append(loss_bmi_l1.item())
            writer.add_scalar('PreTraining/BMIL1Loss',
                              np.array(loss_bmi_l1.item()),
                              train_iter)

            train_sex_ce_loss.append(loss_sex_ce.item())
            writer.add_scalar('PreTraining/SexCELoss',
                              np.array(loss_sex_ce.item()),
                              train_iter)
            train_sex_acc.append(sex_acc.item())
            writer.add_scalar('PreTraining/SexAcc',
                              np.array(sex_acc.item()),
                              train_iter)
            if train_iter > 1000:
                return 0
        
        # Validation
        model_F.eval()
        val_loss_psg = []
        val_age_l1_loss_psg = []
        val_bmi_l1_loss_psg = []
        val_sex_ce_loss_psg = []
        val_sex_acc_psg = []
        with torch.no_grad():
            for X, y, z in val_gen:
                val_loss = 0.0
                val_age_l1_loss = 0.0
                val_bmi_l1_loss = 0.0
                val_sex_ce_loss = 0.0
                val_sex_acc = 0.0
                X, y, z = torch.squeeze(X, 0), torch.squeeze(y, 0), torch.squeeze(z, 0)
                val_epoch_set = PSG_epoch_Dataset(config, X, y, z)
                val_epoch_gen = DataLoader(val_epoch_set, **test_epoch_params)
                for psg, lab, _ in val_epoch_gen:
                    psg, lab = psg.to(device), lab.to(device)
                    out_F, _ = model_F(psg)
                    loss_F, loss_age_l1, loss_bmi_l1, loss_sex_ce = loss_fn_F(out_F, lab)
                    sex_acc = accuracy(out_F[:,2:],lab[:,2])

                    val_loss += loss_F.item() * lab.size(0) / float(len(val_epoch_set))
                    val_age_l1_loss += loss_age_l1.item() * lab.size(0) / float(len(val_epoch_set))
                    val_bmi_l1_loss += loss_bmi_l1.item() * lab.size(0) / float(len(val_epoch_set))
                    val_sex_ce_loss += loss_sex_ce.item() * lab.size(0) / float(len(val_epoch_set))
                    val_sex_acc += sex_acc.item() * lab.size(0) / float(len(val_epoch_set))
                    
                val_loss_psg.append(val_loss)
                val_age_l1_loss_psg.append(val_age_l1_loss)
                val_bmi_l1_loss_psg.append(val_bmi_l1_loss)
                val_sex_ce_loss_psg.append(val_sex_ce_loss)
                val_sex_acc_psg.append(val_sex_acc)
        
        # Track loss
        # Mean loss
        writer.add_scalar('PreValidation/MultiLoss', 
                          np.mean(val_loss_psg), 
                          (epoch + 1) * len(train_set) / config.pre_batch_size)
        writer.add_scalar('PreValidation/AgeL1Loss', 
                          np.mean(val_age_l1_loss_psg), 
                          (epoch + 1) * len(train_set) / config.pre_batch_size)
        writer.add_scalar('PreValidation/BMIL1Loss', 
                          np.mean(val_bmi_l1_loss_psg), 
                          (epoch + 1) * len(train_set) / config.pre_batch_size)
        writer.add_scalar('PreValidation/SexCELoss', 
                          np.mean(val_sex_ce_loss_psg), 
                          (epoch + 1) * len(train_set) / config.pre_batch_size)
        writer.add_scalar('PreValidation/SexAcc', 
                          np.mean(val_sex_acc_psg), 
                          (epoch + 1) * len(train_set) / config.pre_batch_size)   
        # Loss histogram
        writer.add_histogram('PreValidation/MultiLoss_H', 
                          np.array(val_loss_psg), 
                          (epoch + 1) * len(train_set) / config.pre_batch_size)
        writer.add_histogram('PreValidation/AgeL1Loss_H', 
                          np.array(val_age_l1_loss_psg), 
                          (epoch + 1) * len(train_set) / config.pre_batch_size)
        writer.add_histogram('PreValidation/BMIL1Loss_H', 
                          np.array(val_bmi_l1_loss_psg), 
                          (epoch + 1) * len(train_set) / config.pre_batch_size)
        writer.add_histogram('PreValidation/SexCELoss_H', 
                          np.array(val_sex_ce_loss_psg), 
                          (epoch + 1) * len(train_set) / config.pre_batch_size)
        writer.add_histogram('PreValidation/SexAcc_H', 
                          np.array(val_sex_acc_psg), 
                          (epoch + 1) * len(train_set) / config.pre_batch_size)   


        val_loss = np.mean(val_loss_psg)
        print('Train Epoch: {} of {}, Training loss: {:.6f}, Val loss: {:.6f}'.format(
                    epoch+1, config.pre_max_epochs, np.mean(train_loss), val_loss))
    
        # Save Model
        if epoch == 0:
            best_loss = val_loss
            best_epoch = epoch
            best_net = copy.deepcopy(model_F)
        elif val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_net = copy.deepcopy(model_F)
        save_model_F(config.model_F_path, epoch, model_F, optimizer_F, val_loss, best_net, best_epoch, best_loss)
    
    return best_net

def train(args, config, model_L, device, writer):
    
    # Parameters
    train_params = {'batch_size': config.batch_size,
                       'shuffle': True,
                       'num_workers': config.n_workers}
    
    # Generators
    train_set = PSG_feature_Dataset(config, 'train_F')
    train_gen = DataLoader(train_set, **train_params)
    val_set = PSG_feature_Dataset(config, 'val_F')
    val_gen = DataLoader(val_set, **train_params)
    
    # Optimization
    optimizer_L = torch.optim.Adam(model_L.parameters(), lr=config.lr, weight_decay = config.l2)
    age_loss_weight = 68.13
    loss_fn_L = lambda y, t: HuberLoss(y, t, 5, age_loss_weight, device)
    L1Loss = nn.L1Loss()
    
    # Continue previous pretraining
    if not args.train:
        if args.load_best_train:
            checkpoint = torch.load(config.model_L_path + '.tar')
            model_L = checkpoint['best_model']
        return model_L
    elif args.train_continue:
        checkpoint = torch.load(config.model_L_path + '.tar')
        start_epoch = checkpoint['epoch']
        model_L.load_state_dict(checkpoint['model_state_dict'])
        optimizer_L.load_state_dict(checkpoint['optimizer_state_dict'])
        train_iter = start_epoch * len(train_set) // config.batch_size
        best_loss = checkpoint['best_loss']
        
    else:
        train_iter = 0
        checkpoint = 0
        start_epoch = 0
    
    # Loop over epochs
    train_loss_all = []
    train_l1_loss_all = []
    val_loss_all = []
    val_l1_loss_all = []
    
    for epoch in range(start_epoch, config.max_epochs):
        # Training
        train_loss = []
        train_l1_loss = []
        model_L.train()
        for feat, lab, z in train_gen:
            # Zero Gradients
            optimizer_L.zero_grad()
            # Make predictions
            feat, lab, z = feat.to(device), lab.to(device), z.to(device)
            out_L = model_L(feat, z)
            # Optimize network
            loss_L = loss_fn_L(out_L, lab[:,0])
            loss_L.backward()
            optimizer_L.step()
            # Track the loss
            train_loss.append(loss_L.item())
            train_iter += 1
            loss_l1 = L1Loss(out_L.type(torch.float), lab[:, 0].type(torch.float))
            train_l1_loss.append(loss_l1.item())
            writer.add_scalar('Training/L1Loss', 
                              np.array(loss_l1.item()), 
                              train_iter)
            writer.add_scalar('Training/HuberLoss', 
                              np.array(loss_L.item()), 
                              train_iter)
        
        # Validation
        model_L.eval()
        val_loss = 0.0
        val_l1_loss = 0.0
        with torch.no_grad():
            for feat, lab, z in val_gen:
                feat, lab, z = feat.to(device), lab.to(device), z.to(device)
                out_L = model_L(feat, z)
                val_loss += loss_fn_L(out_L, lab[:,0]).item() * lab.size(0) / float(len(val_set))
                val_l1_loss += L1Loss(out_L.type(torch.float), lab[:,0].type(torch.float)).item() * lab.size(0) / float(len(val_set))
                    
        train_loss_all.extend(train_loss)
        train_l1_loss_all.extend(train_l1_loss)
        val_loss_all.append(val_loss)
        val_l1_loss_all.append(val_l1_loss)
        
        # Track loss
        writer.add_scalar('Validation/L1Loss', 
                          val_l1_loss, 
                          (epoch + 1) * len(train_set) / config.batch_size)
        writer.add_scalar('Validation/HuberLoss', 
                          val_loss, 
                          (epoch + 1) * len(train_set) / config.batch_size)
    
        print('Train Epoch: {} of {}, Training loss: {:.6f}, Val loss: {:.6f}'.format(
                    epoch+1, config.max_epochs, np.mean(train_loss), val_loss))
    
        # Save Model
        if epoch == 0:
            best_loss = val_loss
            best_epoch = epoch
            best_net = copy.deepcopy(model_L)
        elif val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_net = copy.deepcopy(model_L)
        save_model_F(config.model_L_path, epoch, model_L, optimizer_L, val_loss, best_net, best_epoch, best_loss)
    
    return model_L, best_loss

def pre_test(args, config, model_F, device, writer):
    
    test_params = {'batch_size': 1,
                   'shuffle': False,
                   'num_workers': config.pre_n_workers}
    pretrain_params = {'batch_size': config.pre_batch_size,
                       'shuffle': False,
                       'num_workers': config.pre_n_workers}
    
    loss_fn_F = multi_label_loss(device)
    
    train_set = PSG_pretrain_Dataset(config, 'train')
    test_set = PSG_Dataset(config, 'test')
    test_gen = DataLoader(test_set, **test_params)
    
    # Testing
    test_loss = 0.0
    test_age_l1_loss = 0.0
    test_bmi_l1_loss = 0.0
    test_sex_ce_loss = 0.0
    test_sex_acc = 0.0
    test_loss_avg = 0.0
    test_age_l1_loss_avg = 0.0
    test_bmi_l1_loss_avg = 0.0
    test_sex_ce_loss_avg = 0.0
    test_sex_acc_avg = 0.0
    target_epoch = []
    pred_epoch = []
    target_avg = []
    pred_avg = []
    model_F.eval()
    with torch.no_grad():
        for X, y, z in test_gen:
            X, y, z = torch.squeeze(X, 0), torch.squeeze(y, 0), torch.squeeze(z, 0)
            test_epoch_set = PSG_epoch_Dataset(config, X, y, z)
            test_epoch_gen = DataLoader(test_epoch_set, **pretrain_params)
            output_F_class = []
            for psg, lab, _ in test_epoch_gen:
                psg, lab = psg.to(device), lab.to(device)
                out_F, _ = model_F(psg)
                loss_F, loss_age_l1, loss_bmi_l1, loss_sex_ce = loss_fn_F(out_F, lab)
                sex_acc = accuracy(out_F[:,2:],lab[:,2])
                test_loss += loss_F.item() * lab.size(0) / float(len(test_epoch_set))
                test_age_l1_loss += loss_age_l1.item() * lab.size(0) / float(len(test_epoch_set))
                test_bmi_l1_loss += loss_bmi_l1.item() * lab.size(0) / float(len(test_epoch_set))
                test_sex_ce_loss += loss_sex_ce.item() * lab.size(0) / float(len(test_epoch_set))
                test_sex_acc += sex_acc.item() * lab.size(0) / float(len(test_epoch_set))
                output_F_class.append(out_F)
                
                target_epoch.append(lab.cpu().numpy())
                pred_epoch.append(out_F.cpu().numpy())
            
            target_avg.append(y.cpu().numpy())
            pred_avg.append(torch.mean(torch.cat(output_F_class),0).cpu().numpy())
            y = torch.unsqueeze(y, 0)
            loss_F, loss_age_l1, loss_bmi_l1, loss_sex_ce = loss_fn_F(torch.mean(torch.cat(output_F_class),0,keepdim=True), y.to(device))
            sex_acc = accuracy(torch.mean(torch.cat(output_F_class),0,keepdim=True)[:, 2:], y[:, 2].to(device))
            test_loss_avg += loss_F.item()
            test_age_l1_loss_avg += loss_age_l1.item()
            test_bmi_l1_loss_avg += loss_bmi_l1.item()
            test_sex_ce_loss_avg += loss_sex_ce.item()
            test_sex_acc_avg += sex_acc.item()

            
    test_loss /= float(len(test_set))
    test_age_l1_loss /= float(len(test_set))
    test_bmi_l1_loss /= float(len(test_set))
    test_sex_ce_loss /= float(len(test_set))
    test_sex_acc /= float(len(test_set)) 
    test_loss_avg /= float(len(test_set))
    test_age_l1_loss_avg /= float(len(test_set))
    test_bmi_l1_loss_avg /= float(len(test_set))
    test_sex_ce_loss_avg /= float(len(test_set))
    test_sex_acc_avg /= float(len(test_set))
    
    # Track loss
    writer.add_scalar('PreTest/HuberLoss', 
                      test_loss, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_scalar('PreTest/AgeL1Loss', 
                      test_age_l1_loss, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_scalar('PreTest/BMIL1Loss', 
                      test_bmi_l1_loss, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_scalar('PreTest/SexCELoss', 
                      test_sex_ce_loss, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_scalar('PreTest/SexAcc', 
                      test_sex_acc, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_scalar('PreTest/HuberLossAvg', 
                      test_loss_avg, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_scalar('PreTest/AgeL1LossAvg', 
                      test_age_l1_loss_avg, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_scalar('PreTest/BMIL1LossAvg', 
                      test_bmi_l1_loss_avg, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_scalar('PreTest/SexCELossAvg', 
                      test_sex_ce_loss_avg, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_scalar('PreTest/SexAccAvg', 
                      test_sex_acc_avg, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_figure('age epoch scatter plot', 
                      plot_regression_pred(np.concatenate(pred_epoch)[:,0],np.concatenate(target_epoch)[:,0],'age'), 
                      global_step = (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_figure('age avg scatter plot', 
                      plot_regression_pred(np.stack(pred_avg)[:,0], np.stack(target_avg)[:,0],'age'), 
                      global_step = (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_figure('bmi epoch scatter plot', 
                      plot_regression_pred(np.concatenate(pred_epoch)[:,1],np.concatenate(target_epoch)[:,1],'bmi'), 
                      global_step = (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_figure('bmi avg scatter plot', 
                      plot_regression_pred(np.stack(pred_avg)[:,1], np.stack(target_avg)[:,1],'bmi'), 
                      global_step = (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)

    
    print('Test loss: {:.6f}. Test loss avg: {:.6f}'.format(test_loss, test_loss_avg))
    
    return

def test(args, config, model_L, device, writer):
    
    # Parameters
    train_params = {'batch_size': config.batch_size,
                       'shuffle': False,
                       'num_workers': config.n_workers}
    
    loss_fn_L = lambda y, t: HuberLoss(y, t, 5, 20, device)
    L1Loss = nn.L1Loss()
    
    train_set = PSG_feature_Dataset(config, 'train_F')
    test_set = PSG_feature_Dataset(config, 'test_F')
    test_gen = DataLoader(test_set, **train_params)
    
    # Testing
    test_loss = 0.0
    test_l1_loss = 0.0
    target_full = []
    pred_full = []
    model_L.eval()
    with torch.no_grad():
        for feat, lab, z in test_gen:
            feat, lab, z = feat.to(device), lab.to(device), z.to(device)
            out_L = model_L(feat, z)
            test_loss += loss_fn_L(out_L, lab[:,0]).item() * lab.size(0) / float(len(test_set))
            test_l1_loss += L1Loss(out_L.type(torch.float), lab[:,0].type(torch.float)).item() * lab.size(0) / float(len(test_set))
            pred_full.append(out_L.cpu().numpy())
            target_full.append(lab[:,0].cpu().numpy())
    
    # Track loss
    writer.add_scalar('Test/L1Loss', 
                      test_l1_loss, 
                      (config.max_epochs + 1) * len(train_set) / config.batch_size)
    writer.add_scalar('Test/HuberLoss', 
                      test_loss, 
                      (config.max_epochs + 1) * len(train_set) / config.batch_size)
    writer.add_figure('age scatter plot', 
                      plot_regression_pred(np.concatenate(pred_full),np.concatenate(target_full),'age'), 
                      global_step = (config.max_epochs + 1) * len(train_set) / config.batch_size)
    
    print('Test loss: {:.6f}.'.format(test_loss))
    
    return

def train_eval(args, config, model_L, device, writer, params):

    hyper_param_string = 'nAge_dof_075_dol_{:.3f}_lr_{:.5f}_l2_{:.7f}'.format(params['do'], params['lr'], params['l2'])
    
    config.do_l = params['do']
    config.lr = params['lr']
    config.l2 = params['l2']
    config.model_L_path = config.model_L_BO_path + hyper_param_string
    writer = SummaryWriter('runs/' + hyper_param_string)

    model_L, perf = train(args, config, model_L, device, writer) 

    print('Loss: {:.6f}. Dropout: {:.3f}. Learning rate: {:.5f}. L2 weight decay: {:.7f}'.format(perf, params['do'], params['lr'], params['l2']))

    return perf

def bayesian_opt_eval(args, config, model_L, device, writer):

    exp = ax.SimpleExperiment(
        name='age_label_experiment',
        search_space=ax.SearchSpace(
            parameters=[
                ax.RangeParameter(name='do', lower=0.0, upper=0.99, parameter_type=ax.ParameterType.FLOAT),
                ax.RangeParameter(name='lr', lower=10**(-6), upper=10**(-2), parameter_type=ax.ParameterType.FLOAT, log_scale=True),
                ax.RangeParameter(name='l2', lower=10**(-10), upper=10**(-2), parameter_type=ax.ParameterType.FLOAT, log_scale=True),
            ]
        ),
        evaluation_function=lambda p: train_eval(args, config, model_L, device, writer, p),
        minimize=True
    )

    sobol = ax.Models.SOBOL(exp.search_space)
    for i in range(5):
        exp.new_trial(generator_run=sobol.gen(1))

    best_arm = None
    for i in range(15):
        gpei = ax.Models.GPEI(experiment=exp, data=exp.eval())
        generator_run = gpei.gen(1)
        best_arm, _ = generator_run.best_arm_predictions
        exp.new_trial(generator_run=generator_run)

    best_parameters = best_arm.parameters
    ax.save(exp, config.BO_expe_path + '.json')
    return best_parameters

def main(args):
    # Training configurations
    config = Config()
    # Setup training
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Age distribution
    # get_age_label_distribution(config)
    # Example plot
    # plot_train_epoch(config)
    # Loss function plot
    #plot_huber_loss(r = [-20, 20], d = 5, s = 20)
    
    # Model initialization
    model_F = M_PSG2FEAT(config).to(device)
    model_L = M_FEAT2LABEL(config).to(device)
    # Model Debug
    debug_model(model_F, (32, 12, 128*5*60), 2, device)
    debug_model(model_L, (64, 120, 256), 1, device, cond_size = sum(config.label_cond_size))
    # Test num workers
    # analyze_num_workers(config, model_F, 2, device, PSG_pretrain_Dataset(config, 'train'), config.pre_batch_size, n_iter = -1, n_workers_range = np.arange(12))
    # Writer
    writer_name = 'runs/nAge_specL_cond_all_do_75_l2_1e-5_chd_aug_' + time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(writer_name)
    # Pre-Training
    model_F = pretrain(args, config, model_F, device, writer)
    # Pre-testing
    if args.test_pre:
        pre_test(args, config, model_F, device, writer)
    # Save output features
    if args.save_feat:
        save_train_features(config, model_F, device)
    # Training
    if args.bo:
        params = bayesian_opt_eval(args, config, model_L, device, writer)
        config.do_l = params['do']
        config.lr = params['lr']
        config.l2 = params['l2']
        print('Optimal hyperparameters: ', params)
    #model_L, _ = train(args, config, model_L, device, writer)
    # Testing
    if args.test:
        test(args, config, model_L, device, writer)
    
    writer.close()

if __name__ == "__main__":
    main(args)
    
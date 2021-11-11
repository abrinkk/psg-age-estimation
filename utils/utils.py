import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import h5py
import torch
from psg_reader import plot_edf_data
import time
from scipy.stats import pearsonr
import torch.nn as nn
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
from psg_dataset import PSG_Dataset, PSG_epoch_Dataset, PSG_pretrain_Dataset

def HuberLoss(y, t, d, s, device):
    dif = (t.type(torch.FloatTensor) - y.type(torch.FloatTensor)).abs_()
    mask_l2 = (dif < d)
    L = (d * dif - 1/2 * d**2) / s
    L[mask_l2] = (1/2 * dif[mask_l2]**2) / s
    return L.mean().to(device)

def accuracy(y, t):
    y_max = torch.argmax(y, 1)
    return (y_max == t).type(torch.float).mean()

def save_train_features(config, model_F, device):
    
    test_params = {'batch_size': 1,
                   'shuffle': False,
                   'num_workers': config.pre_n_workers}
    pretrain_params = {'batch_size': config.pre_batch_size,
                       'shuffle': False,
                       'num_workers': config.pre_n_workers}
    
    train_set = PSG_Dataset(config, 'train', return_filename = True)
    train_gen = DataLoader(train_set, **test_params)
    val_set = PSG_Dataset(config, 'val', return_filename = True)
    val_gen = DataLoader(val_set, **test_params)
    test_set = PSG_Dataset(config, 'test', return_filename = True)
    test_gen = DataLoader(test_set, **test_params)
    
    # Generating and saving features
    model_F.eval()
    with torch.no_grad():
        for X, y, z, filename in train_gen:
            filesplit = os.path.split(filename[0])
            output_filename = filesplit[0] + '_F\\' + filesplit[1]
            X, y, z = torch.squeeze(X, 0), torch.squeeze(y, 0), torch.squeeze(z, 0)
            test_epoch_set = PSG_epoch_Dataset(config, X, y, z)
            test_epoch_gen = DataLoader(test_epoch_set, **pretrain_params)
            # Collect faetures
            output_F = []
            for psg, lab, z_e in test_epoch_gen:
                psg, lab, z_e = psg.to(device), lab.to(device), z_e.to(device)
                _, output_F_b = model_F(psg)
                output_F.append(output_F_b)
            output_F_psg = torch.cat(output_F, 0).detach().cpu().numpy()
            with h5py.File(output_filename, "w") as f:
                f.create_dataset("PSG", data = output_F_psg, dtype='f4')
                for idx, i in enumerate(config.label):
                    f.attrs[i] = y[0].cpu().numpy()
                cond_idx_start = 0
                for idx, i in enumerate(config.label_cond):
                    cond_idx_end = cond_idx_start + config.label_cond_size[idx]
                    f.attrs[i] = z[cond_idx_start:cond_idx_end].cpu().numpy()
                    cond_idx_start = cond_idx_end
    
    with torch.no_grad():
        for X, y, z, filename in val_gen:
            filesplit = os.path.split(filename[0])
            output_filename = filesplit[0] + '_F\\' + filesplit[1]
            X, y, z = torch.squeeze(X, 0), torch.squeeze(y, 0), torch.squeeze(z, 0)
            test_epoch_set = PSG_epoch_Dataset(config, X, y, z)
            test_epoch_gen = DataLoader(test_epoch_set, **pretrain_params)
            # Collect faetures
            output_F = []
            for psg, lab, z_e in test_epoch_gen:
                psg, lab, z_e = psg.to(device), lab.to(device), z_e.to(device)
                _, output_F_b = model_F(psg)
                output_F.append(output_F_b)
            output_F_psg = torch.cat(output_F, 0).detach().cpu().numpy()
            with h5py.File(output_filename, "w") as f:
                f.create_dataset("PSG", data = output_F_psg, dtype='f4')
                for idx, i in enumerate(config.label):
                    f.attrs[i] = y[0].cpu().numpy()
                cond_idx_start = 0
                for idx, i in enumerate(config.label_cond):
                    cond_idx_end = cond_idx_start + config.label_cond_size[idx]
                    f.attrs[i] = z[cond_idx_start:cond_idx_end].cpu().numpy()
                    cond_idx_start = cond_idx_end
    
    
    with torch.no_grad():
        for X, y, z, filename in test_gen:
            filesplit = os.path.split(filename[0])
            output_filename = filesplit[0] + '_F\\' + filesplit[1]
            X, y, z = torch.squeeze(X, 0), torch.squeeze(y, 0), torch.squeeze(z, 0)
            test_epoch_set = PSG_epoch_Dataset(config, X, y, z)
            test_epoch_gen = DataLoader(test_epoch_set, **pretrain_params)
            # Collect faetures
            output_F = []
            for psg, lab, z_e in test_epoch_gen:
                psg, lab, z_e = psg.to(device), lab.to(device), z_e.to(device)
                _, output_F_b = model_F(psg)
                output_F.append(output_F_b)
            output_F_psg = torch.cat(output_F, 0).detach().cpu().numpy()
            with h5py.File(output_filename, "w") as f:
                f.create_dataset("PSG", data = output_F_psg, dtype='f4')
                for idx, i in enumerate(config.label):
                    f.attrs[i] = y[0].cpu().numpy()
                cond_idx_start = 0
                for idx, i in enumerate(config.label_cond):
                    cond_idx_end = cond_idx_start + config.label_cond_size[idx]
                    f.attrs[i] = z[cond_idx_start:cond_idx_end].cpu().numpy()
                    cond_idx_start = cond_idx_end
    return

def plot_preds(pred, lab, loss):
    '''
    Generates a matplotlib Figure that shows bar plots of predicted label 
    probabilities along with the true label.
    '''
    fig = plt.figure(figsize = (15,5))
    n = pred.shape[0]
    for idx in np.arange(n):
        ax = fig.add_subplot(1, n, idx + 1, xticks = [1, 15, 30, 45, 60, 75, 90])
        ax.bar(np.arange(1,91), pred[idx])
        ax.bar(lab[idx], pred[idx, lab[idx] + 1])
        ax.set_ylabel('prediction')
        ax.set_xlabel('age')
        ax.set_title('Loss: {:.6f}'.format(loss[idx]))
    
    return fig

def plot_regression_pred(pred, target, variable_name):
    pred = np.array(pred)
    target = np.array(target)
    age_corr = pearsonr(target, pred)
    age_text = 'r = %.3f' % (age_corr[0])
    fig = plt.figure(figsize = (5,5), dpi = 300)
    ax = sns.scatterplot(x = target, y = pred)
    ax.plot([0, 100], [0, 100],'--r', alpha = 0.5)
    if variable_name == 'age':
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        ax.set_xlabel('Chronological Age')
        ax.set_ylabel('Sleep Age')
        ax.text(15, 85, age_text, fontsize=12)
    elif variable_name == 'bmi':
        ax.set_xlim([10, 70])
        ax.set_ylim([10, 70])
        ax.set_xlabel('True BMI')
        ax.set_ylabel('Sleep BMI')
        ax.text(20, 55, age_text, fontsize=12)
    return fig

def get_age_label_distribution(config):
    
    # Parameters
    test_params = {'batch_size': 1,
                   'shuffle': False,
                   'num_workers': config.pre_n_workers}
    
    # Generators
    train_set = PSG_Dataset(config, 'train')
    train_gen = DataLoader(train_set, **test_params)
    val_set = PSG_Dataset(config, 'val')
    val_gen = DataLoader(val_set, **test_params)
    test_set = PSG_Dataset(config, 'test')
    test_gen = DataLoader(test_set, **test_params)
    
    y_train = []
    y_val = []
    y_test  = []
    for _, y in train_gen:
        y_train.append(y.numpy())
    for _, y in val_gen:
        y_val.append(y.numpy())
    for _, y in test_gen:
        y_test.append(y.numpy())
    
    fig, axes = plt.subplots(3,1, figsize = (10,5), dpi = 300)
    sns.distplot(y_train, bins = np.arange(0, 90, 1), kde = False, ax = axes[0])
    sns.distplot(y_val, bins = np.arange(0, 90, 1), kde = False, ax = axes[1])
    sns.distplot(y_test, bins = np.arange(0, 90, 1), kde = False, ax = axes[2])
    axes[0].set_title('Train set')
    axes[0].set_xticks([0, 15, 30, 45, 60, 75, 90])
    axes[0].set_xticklabels(['']*7)
    axes[1].set_title('Validation set')
    axes[1].set_xticks([0, 15, 30, 45, 60, 75, 90])
    axes[1].set_xticklabels(['']*7)
    axes[2].set_title('Test set')
    axes[2].set_xticks([0, 15, 30, 45, 60, 75, 90])
    axes[2].set_xlabel('Chonological Age')
    plt.show()
    return
    

def save_model_F(filename, epoch, model, optimizer, loss, net, best_epoch, best_loss):
    save_dict = {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': loss,
                 'best_model': net,
                 'best_epoch': best_epoch, 
                 'best_loss': best_loss}
    torch.save(save_dict, filename + '.tar')
    return

def save_model_L(filename, epoch, model_F, model_L, optimizer_F, optimizer_L, loss, net_F, net_L, best_epoch, best_loss):
    save_dict = {'epoch': epoch,
                 'model_F_state_dict': model_F.state_dict(),
                 'optimizer_F_state_dict': optimizer_F.state_dict(),
                 'model_L_state_dict': model_F.state_dict(),
                 'optimizer_L_state_dict': optimizer_L.state_dict(),
                 'loss': loss,
                 'best_model_F': net_F,
                 'best_model_L': net_L,
                 'best_epoch': best_epoch, 
                 'best_loss': best_loss}
    torch.save(save_dict, filename + '.tar')
    return

def analyze_num_workers(config, model, num_out, device, dataset_a, batch_size, n_iter = -1, n_workers_range = np.arange(12)):
    loss_fn_F = lambda y, t: HuberLoss(y, t, 5, 20, device)
    for n_workers in n_workers_range:
        params = {'batch_size': batch_size,
                       'shuffle': True,
                       'num_workers': n_workers}
        gen_a = DataLoader(dataset_a, **params)
        
        time_start = time.time()
        n = 0
        for psg, lab, _ in gen_a:
            # Zero Gradients
            model.zero_grad()
            # Make predictions
            psg, lab = psg.to(device), lab.to(device)
            if num_out == 2:
                out_F, _ = model(psg)
            else:
                out_F = model(psg)
            # Optimize network
            loss_F = loss_fn_F(out_F, lab)
            loss_F.backward()
            n += 1
            if n_iter != -1 and n > n_iter:
                break

        print('n_workers: {:.1f}, epoch time: {:.3f}'.format(n_workers, time.time() - time_start))
    return

def debug_model(model, input_size, output_N, device, cond_size = False):
    if not cond_size:
        model.summary(input_size[1:], device, input_size[0])
    else:
        model.summary([input_size[1:], (cond_size,)], device, input_size[0])
        z = torch.rand((input_size[0],cond_size)).to(device)
    X = torch.rand(input_size).to(device)
    print('Input size: ',X.size)
    time_start = time.time()
    if not cond_size:
        out = model(X)
    else:
        out = model(X, z)
    print('Batch time: {:.3f}'.format(time.time() - time_start))
    if output_N > 1:
        for i in out:
            print('Output size: ',i.size())
    else:
        print('Output size: ',out.size())
            
    return

def plot_train_epoch(config):
    test_params = {'batch_size': 1,
                   'shuffle': False,
                   'num_workers': config.pre_n_workers}
    
    train_set = PSG_Dataset(config, 'train')
    train_gen = DataLoader(train_set, **test_params)
    train_iter = iter(train_gen)
    X, y = next(train_iter)
    psg = X[0,21].numpy()
    plot_epoch(psg)
    return

def plot_epoch(psg):
    channels = ['C3','C4','EOGL','EOGR','ECG','Chin','Leg','Airflow','NasalP','Abd','Chest','OSat']
    fs = [128]*len(channels)
    plot_edf_data({'x': psg, 'fs': fs}, channels, save_fig=True)
    return


def plot_huber_loss(r = [-20, 21], d = 5, s = 20):
    pred = np.arange(r[0],r[1])
    target = np.zeros_like(pred)
    dif = np.abs(pred - target)
    mask = dif < d
    L = (d * dif - 1/2 * d**2) / s
    L[mask] = (1/2 * dif[mask]**2) / s
    
    plt.figure(figsize = (5,5), dpi = 300)
    plt.plot(pred, L)
    plt.xlim(r)
    plt.xlabel('(Pred - Target)')
    plt.ylabel('Scaled Huber Loss')
    plt.show()
    
    return
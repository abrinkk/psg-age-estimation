import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import h5py
import torch
import time
from scipy.stats import pearsonr
import torch.nn as nn

def HuberLoss(y, t, d, s, device):
    dif = (t.type(torch.float) - y.type(torch.float)).abs_()
    mask_l2 = (dif < d)
    L = (d * dif - 1/2 * d**2) / s
    L[mask_l2] = (1/2 * dif[mask_l2]**2) / s
    return L.mean().to(device)

def HuberLoss_clamped(y, t, d, s, device):
    clamp_level = 60
    dif = (t.type(torch.float) - y.type(torch.float))
    dif_a = torch.abs(dif)
    mask_l2 = (dif_a < d)
    mask_zero = ((dif < 0) & (t > clamp_level)) | ((dif > 0) & (t < clamp_level)) & mask_l2
    L = (d * dif_a - 1/2 * d**2) / s
    L[mask_l2] = (1/2 * dif_a[mask_l2]**2) / s
    L[mask_zero] = 0
    return L.mean().to(device)

def HuberLoss_flat_0(y, t, d, s, device):
    dif = (t.type(torch.float) - y.type(torch.float)).abs_()
    mask_l2 = (dif < d*3/2)
    mask_0 = (dif < d/2)
    L = (d * dif - 1/2 * d**2) / s
    L[mask_l2] = (1/2 * dif[mask_l2]**2) / s
    L[mask_0] = 0
    return L

def HuberLoss_flat(y, t, d, s, device):
    clamp_level = 60
    mask_t_upper = t > clamp_level
    mask_t_lower = t <= clamp_level
    L_t_upper = HuberLoss_flat_0(y[mask_t_upper] - d/2, t[mask_t_upper], d, s, device)
    L_t_lower = HuberLoss_flat_0(y[mask_t_lower] + d/2, t[mask_t_lower], d, s, device)
    L = (L_t_upper.sum() + L_t_lower.sum()) / y.size(0)
    return L.to(device)
    
def accuracy(y, t):
    y_max = torch.argmax(y, 1)
    return (y_max == t).type(torch.float).mean()

def cov_loss(y, t, gamma_cov, device):
    dif = (y.type(torch.float) - t.type(torch.float)).to(device)
    vx  = dif - torch.mean(dif)
    vy  = t - torch.mean(t)
    # TODO: Clamp before sum
    cov_prod = vx * vy
    cov_prod = torch.clamp(-cov_prod, min = 0)
    cov = torch.sum(cov_prod) / (vx.size(0) - 1)
    # cov = torch.sum(vx * vy) / (vx.size(0) - 1)
    # cov = torch.clamp(-cov, min = 0)
    return gamma_cov*cov.to(device)

def corr_loss(y, t, beta_corr, device):
    eps = 1e-10
    dif = (y.type(torch.float) - t.type(torch.float)).to(device)
    vx  = dif - torch.mean(dif)
    vy  = t - torch.mean(t)
    sigma_x = torch.sqrt(eps + torch.sum(vx ** 2))
    sigma_y = torch.sqrt(eps + torch.sum(vy ** 2))
    cov = torch.sum(vx * vy)
    cor = cov / (eps + sigma_x * sigma_y)
    cor = torch.clamp(cor, -1.0, 1.0)
    return 0.5*torch.pow(cor.abs_(), beta_corr).to(device)

def nll_loss(target, mu, pdf_shape, dist_type = 'normal'):
    # Using positive-definite function on pdf_shape 
    # TODO: test clamp, exp and others
    #pdf_shape = torch.exp(pdf_shape).unsqueeze(1)
    pdf_shape = torch.clamp(pdf_shape, 0.1).unsqueeze(1)
    mu = mu.unsqueeze(1)
    if dist_type == 'normal':
        dist = torch.distributions.normal.Normal(mu, pdf_shape)
    elif dist_type == 'gamma':
        # input alpha, beta
        dist = torch.distributions.gamma.Gamma(pdf_shape, pdf_shape / mu)

    # Compute nll
    return torch.mean(- dist.log_prob(target))


class triplet_loss(nn.Module):
    def __init__(self, bs, n_neg):
        super(triplet_loss, self).__init__()
        assert bs - 1 >= n_neg
        self.bs = bs
        self.K = n_neg

    def forward(self, z_pos, z_neg):
        '''
        z_pos:  representation of x_pos, a sub-sequence of x_neg 
                z_pos.size() = [bs, z_size]
        z_neg:  representation of x_neg
                z_neg.size() = [bs, z_size]
        '''
        z_size = z_pos.size(1)

        # positive loss
        loss_pos = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(z_neg.view(self.bs, 1, z_size), z_pos.view(self.bs, z_size, 1))), [1, 2])

        # random negative samples from other batches
        neg_samples = torch.LongTensor(np.concatenate([np.random.choice(np.concatenate([np.arange(1, i), np.arange(i + 1, self.bs)]), self.K) for i in range(1, self.bs + 1)], 0))

        # negative loss
        loss_neg = -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(z_neg.view(self.bs, 1, z_size), z_neg[neg_samples].view(self.bs, self.K, z_size).permute(0, 2, 1))), [1, 2])

        # combine loss
        loss = torch.mean(loss_pos + loss_neg)

        return loss

class age_loss(nn.Module):
    '''
    loss over (age)
    '''
    def __init__(self, device, loss_method = 'huber', gamma_cov = 0.01):
        super(age_loss, self).__init__()
        self.eps = 1e-10
        self.L1Loss = nn.L1Loss()
        self.L2Loss = nn.MSELoss()
        self.device = device
        self.gamma_cov = gamma_cov
        self.loss_method = loss_method
        # Correction of loss size guess 50 with target 75
        if loss_method == 'huber':
            self.multi_weights = 112.5
        elif loss_method == 'l1':
            self.multi_weights = 25
        elif loss_method == 'l2':
            self.multi_weights = 25*25
        elif loss_method == 'nll_normal':
            self.multi_weights = 6.3465
        elif loss_method == 'nll_gamma':
            self.multi_weights = 5.0388

    def forward(self, y, t, pdf_shape = None):

        # L1 age
        loss_age_l1 = self.L1Loss(y, t[:, 0])

        # Huber-loss
        if self.loss_method == 'huber':
            loss_age = HuberLoss(y, t[:, 0], 5, self.multi_weights, self.device)
        elif self.loss_method == 'l1':
            loss_age = loss_age_l1 / self.multi_weights
        elif self.loss_method == 'l2':
            loss_age = self.L2Loss(y, t[:, 0]) / self.multi_weights
        elif self.loss_method == 'nll_normal':
            loss_age = nll_loss(t, y, pdf_shape, dist_type='normal') / self.multi_weights
        elif self.loss_method == 'nll_gamma':
            loss_age = nll_loss(t, y, pdf_shape, dist_type='gamma') / self.multi_weights

        # Covariance loss
        loss_cov = cov_loss(y, t[:,0], self.gamma_cov, self.device)

        # Collect losses
        out = {'loss': loss_age,
                'age_l1_loss': loss_age_l1,
                'loss_cov': loss_cov / (self.eps + self.gamma_cov)}
        return out

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

    def forward(self, y, t):

        # L1 age and bmi
        loss_age_l1 = self.L1Loss(y[:, 0], t[:, 0]) # Baseline 15.96
        loss_bmi_l1 = self.L1Loss(y[:, 1], t[:, 1]) # Baseline 5.97

        # Multi Huber-CE-loss
        loss_age = HuberLoss(y[:, 0], t[:, 0], 5, self.multi_weights[0], self.device)
        loss_bmi = HuberLoss(y[:, 1], t[:, 1], 5, self.multi_weights[1], self.device)
        loss_sex = self.CELoss(y[:, 2:], t[:, 2].long()).to(self.device) # Baseline 0.62
        loss_multi = 1/3 * (loss_age + loss_bmi + loss_sex / self.multi_weights[2])

        sex_acc = accuracy(y[:, 2:], t[:, 2])

        # Collect losses
        out = {'loss': loss_multi,
                'age_l1_loss': loss_age_l1,
                'bmi_l1_loss': loss_bmi_l1,
                'sex_ce_loss': loss_sex,
                'sex_acc': sex_acc}
        return out


if __name__ == "__main__":
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # predictions and target
    t_ = np.array([[11.3], [61.8], [9.4], [54.1], [44.6], [81.1], [32.9], [25.5], [75.0], [41.4]])
    p_ = [14.2, 55.1, 20.1, 54.2, 46.1, 71.2, 35.1, 39.1, 77.0, 45.1]
    pdf_shape_ = [10.2, 10.1, 10.1, 15.2, 2.1, 59.2, 2.1, 39.1, 5.0, 45.1]

    # random pred and target
    t_r = np.random.normal(60, 10, 32)
    p_r = np.random.normal(0, 5, 32) + t_r
    pdf_shape_r = np.random.normal(0, 5, 32) + t_r

    # to torch
    t = torch.Tensor(t_).to(device)
    p = torch.Tensor(p_).to(device)
    pdf_shape = torch.Tensor(pdf_shape_).to(device)
    tr = torch.Tensor(t_r).unsqueeze(1).to(device)
    pr = torch.Tensor(p_r).to(device)
    pdf_shaper = torch.Tensor(pdf_shape_r).to(device)

    cov_test = np.cov(t_[:, 0],p_ - t_[:, 0])

    cov_test_r = np.cov(t_r, p_r - t_r)

    loss_fn = age_loss(device, loss_method='nll_normal', gamma_cov=0.01)

    loss_test = loss_fn(p, t, pdf_shape)

    loss_test_r = loss_fn(pr, tr, pdf_shaper)

    print(loss_test, loss_test_r, cov_test, cov_test_r)

    #loss = HuberLoss(p, t[:, 0], 5, 68.13, device)

    #print(p,t,loss)

    z_pos = torch.randn([32, 128]).to(device)
    z_neg = torch.randn([32, 128]).to(device)

    unsupervised_loss = triplet_loss(32, 8)

    print(unsupervised_loss(z_pos, z_neg))
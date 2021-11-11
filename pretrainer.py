import os
import h5py

import numpy as np
import pandas as pd

import torch
from torch.utils import data

from tqdm import tqdm

from base.base_trainer import BaseTrainer
from utils.util_am import am_model

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    InputXGradient,
    IntegratedGradients,
    NoiseTunnel,
)

class PreTrainer(BaseTrainer):
    def __init__(self, network, loss_fn, config, 
                 writer=None,
                 device='cpu',
                 num_epochs=100,
                 patience=None,
                 resume=None,
                 scheduler=None):
        """Class to train an age estimation network

        Args:
            network (M_PSG2FEAT): An network for age estimation in polysomnography epochs
            loss_fn ([type]): a loss function for regression
            config (Config): an instance of the config class with set attributes.
            writer (SummaryWriter, optional): Tensorboard writer. Defaults to None.
            device (str, optional): device for cpu or gpu processing. Defaults to 'cpu'.
            num_epochs (int, optional): maximum number of epochs. Defaults to 100.
            patience (int, optional): early stopping patience. Defaults to None.
            resume (boolean, optional): to resume existing checkpoint. Defaults to None.
            scheduler (None, optional): Learning rate scheduler (not used). Defaults to None.
        """
        # Change config
        config.lr = config.pre_lr
        super().__init__(network, loss_fn, config,
                         device=device,
                         num_epochs=num_epochs,
                         patience=patience,
                         resume=resume,
                         scheduler=scheduler)
        self.writer = writer
        self.iter = 0
        self.len_epoch = 0
        self.nll_loss = True if config.loss_func == 'nll_normal' or config.loss_func == 'nll_gamma' else False

    def train_and_validate(self, train_dataloader, eval_dataloader):
        """train and validation loops

        Args:
            train_dataloader (PSG_pretrain_Dataset): Training dataset
            eval_dataloader (PSG_pretrain_Dataset): Validation dataset
        """
        self.on_begin()
        
        # Length of epoch
        self.len_epoch = len(train_dataloader)

        if self.resume:
            self.restore_checkpoint()
            self.iter = self.current_epoch * self.len_epoch // self.config.pre_batch_size

        for self.current_epoch in range(self.start_epoch, self.num_epochs):

            self.on_begin_epoch()

            # Training phase
            self.network.train()
            batch_losses = []
            bar_train = tqdm(train_dataloader,
                             total=self.len_epoch,
                             desc=f'Loss: {np.inf:.04f}')
            for batch in bar_train:

                # Run single training step
                loss_out = self.train_step(batch)
                if isinstance(loss_out, dict):
                    loss = loss_out['loss'].item()
                else:
                    loss = loss_out.item()
                batch_losses.append(loss)

                # Update bar state
                bar_train.set_description(f'Loss: {loss:.04f}')

                # Log metrics to tensorboard
                self.iter += 1
                self.log_metrics_to_tensorboard(loss_out, 'PreTraining')
                #if self.iter > 1000:
                #    return

            self.train_losses.append(np.mean(batch_losses))

            # Validation phase
            self.network.eval()
            batch_losses = []
            batch_loss_outs = []
            bar_eval = tqdm(eval_dataloader,
                            total=len(eval_dataloader),
                            desc=f'Loss: {np.inf:.04f}')
            with torch.no_grad():
                for batch in bar_eval:

                    # Run single validation step
                    loss_out = self.eval_step(batch)
                    if isinstance(loss_out, dict):
                        loss = loss_out['loss'].item()
                    else:
                        loss = loss_out.item()
                    batch_losses.append(loss)
                    batch_loss_outs.append(loss_out)

                    # Update bar state
                    bar_eval.set_description(f'Loss: {loss:.04f}')

                self.eval_losses.append(np.mean(batch_losses))
            
            # Log metrics to tensorboard
            self.log_metrics_to_tensorboard(batch_loss_outs, 'PreValidation')

            self.on_end_epoch()

            if self.patience_counter > self.patience:
                print('\nEarly stopping criterion reached, stopping training!')
                break

        self.on_end()

        return None

    def train_step(self, batch):
        """Function to call a single training step using a batch

        Args:
            batch (dict): A dict including Tensors for input and labels

        Returns:
            loss (Tensor): Training loss
        """
        x, y = batch['data'].to(self.device), batch['label'].to(self.device)

        # Reset gradients
        self.optimizer.zero_grad()

        # Run forward pass
        if self.network.return_only_pred:
            y_p = self.network(x)
        else:
            out = self.network(x)
            y_p = out['pred']
            if self.nll_loss:
                pdf_shape = out['pdf_shape']

        # Calculate training loss
        if self.nll_loss:
            loss_out = self.loss_fn(y_p, y, pdf_shape)
        else:
            loss_out = self.loss_fn(y_p, y)
        if isinstance(loss_out, dict):
            loss = loss_out['loss']
        else:
            loss = loss_out

        # Check for NaN
        if self.check_nan(y, y_p, loss):
            print('data: ', torch.any(torch.isnan(x)).item())
            print('label: ', y)
            print('pred:  ', y_p)
            print('loss:  ', loss)
            assert not self.check_nan(y, y_p, loss)

        # Run optimization
        loss.backward()
        self.optimizer.step()

        return loss_out

    def eval_step(self, batch):
        """Function to call a single validation step using a batch

        Args:
            batch (dict): A dict including Tensors for input and labels

        Returns:
            loss (Tensor): Training loss
        """
        x, y = batch['data'].to(self.device), batch['label'].to(self.device)

        # Run forward pass
        if self.network.return_only_pred:
            y_p = self.network(x)
        else:
            out = self.network(x)
            y_p = out['pred']
            if self.nll_loss:
                pdf_shape = out['pdf_shape']

        # Calculate training loss
        if self.nll_loss:
            loss_out = self.loss_fn(y_p, y, pdf_shape)
        else:
            loss_out = self.loss_fn(y_p, y)

        return loss_out

    def check_nan(self, y, y_p, loss):
        """Asses that there are no nan values in labels or predictions

        Args:
            y (Tensor): labels
            y_p (Tensor): predictions
            loss (Tensor): loss Tensor 

        Returns:
            (bool): Any nan in y, y_p, or loss
        """
        # Check y
        check_y = any([x.item() != x.item() for x in y])
        # Check y_p
        check_y_p = any([x.item() != x.item() for x in y_p])
        # Check loss
        check_loss = loss.item() != loss.item()
        return any([check_y, check_y_p, check_loss])


    def predict_step(self, batch):
        """Function to call a single prediction step using a batch

        Args:
            batch (dict): A dict including Tensors for input and labels

        Returns:
            out (dict): Dict containing network outputs and batch information
        """
        record, position = batch['fid'], batch['position']
        x, y, z = batch['data'].to(self.device), batch['label'].to(self.device), batch['label_cond'].to(self.device)

        # Run forward pass
        if self.best_network.return_only_pred:
            y_p = self.best_network(x)
            out = {'pred': y_p}
        else:
            out = self.best_network(x)
        
        # Add record and label info
        out['fids'] = record
        out['position'] = position
        out['label'] = y
        out['label_cond'] = z

        return out

    def evaluate_performance(self, test_dataloader, len_epoch = 0):
        """Test performance on test set

        Args:
            test_dataloader (PSG_pretrain_Dataset): A test dataset
            len_epoch (int, optional): Training epoch length, which is used to log performance. Defaults to 0.

        Returns:
            predictions: age predictions per epoch and attention weights
            metrics: a dict containing all record names, ages, average age predictions, and losses
        """
        self.len_epoch = len_epoch

        if self.resume:
            self.restore_checkpoint()
            self.iter = self.last_update * self.len_epoch // self.config.pre_batch_size

        # initialize records
        records = [record for record in test_dataloader.dataset.filenames]
        predictions = {r: {'age_p': [],'pdf_shape': [], 'pos': [], 'label': [], 'alpha': []} for r in records}
        metrics = {'record': records,
                   'age': [],
                   'age_p': [],
                   'loss': [],
                   'age_l1_loss': []}

        print(f'\nEvaluating model')
        self.best_network.eval()
        batch_losses = []
        batch_loss_outs = []
        bar_test = tqdm(test_dataloader, total=len(test_dataloader))
        with torch.no_grad():
            for batch in bar_test:

                fids = batch['fid']
                pos = batch['position']
                y = batch['label'].numpy()
                x = batch['data'].to(self.device)

                # Run forward pass and get predictions
                if self.best_network.return_only_pred:
                    y_p = self.best_network(x)
                else:
                    out = self.best_network(x)
                    y_p = out['pred']
                    pdf_shape = out['pdf_shape']
                    alpha = out['alpha']
                
                # Compute loss
                if self.nll_loss:
                    loss_out = self.loss_fn(y_p, torch.Tensor(y).to(self.device), pdf_shape)
                else:
                    loss_out = self.loss_fn(y_p, torch.Tensor(y).to(self.device))
                if isinstance(loss_out, dict):
                    loss = loss_out['loss'].item()
                else:
                    loss = loss_out.item()
                batch_losses.append(loss)
                batch_loss_outs.append(loss_out)
                
                # Assign to subjects
                for record, r_pos, pred, r_pdf_shape, labels, alpha_w in zip(fids, pos[0], y_p, pdf_shape, y, alpha):
                    # Predictions
                    predictions[record]['pos'].append(r_pos.numpy().tolist())
                    predictions[record]['age_p'].append(pred.cpu().numpy().tolist())
                    predictions[record]['pdf_shape'].append(r_pdf_shape.cpu().numpy().tolist())
                    predictions[record]['label'].append(labels[0])
                    predictions[record]['alpha'].append(alpha_w.cpu().numpy().reshape(-1).tolist())

        # Log to tensorboard
        self.log_metrics_to_tensorboard(batch_loss_outs, 'PreTest')

        # Calculate metrics
        for record in metrics['record']:

            # Label
            y = predictions[record]['label']

            # If no data return NaN
            if len(y) == 0:
                metrics['age'].append(np.NaN)
                metrics['age_p'].append(np.NaN)
                metrics['loss'].append(np.NaN)
                metrics['age_l1_loss'].append(np.NaN)
            else:
                # Average prediction
                y_p = [[np.mean(predictions[record]['age_p'])]]

                # Log labels
                metrics['age'].append(y[0])
                
                # Log averaged predictions
                metrics['age_p'].append(y_p[0][0])

                # Log loss and accuracy
                if self.nll_loss:
                    loss_out = self.loss_fn(torch.Tensor(y_p[0]).to(self.device), torch.Tensor([[y[0]]]).to(self.device), 10*torch.ones_like(torch.Tensor(y_p[0]).to(self.device)))
                else:
                    loss_out = self.loss_fn(torch.Tensor(y_p[0]).to(self.device), torch.Tensor([[y[0]]]).to(self.device))
                metrics['loss'].append(loss_out['loss'].item())
                metrics['age_l1_loss'].append(loss_out['age_l1_loss'].item())

        return pd.DataFrame.from_dict(metrics).set_index('record'), predictions

    def save_features(self, dataloaders):
        """Iterates dataloaders and saves latent space representations of each polysomnography epoch.

        Args:
            dataloaders (dict): A dict containing PSG_pretrain_Dataset instances
        """
        if self.resume:
            self.restore_checkpoint()

        print(f'\nEvaluating features')
        self.best_network.eval()

        # Iterate dataloaders
        for k, dl in dataloaders.items():

            print(f'\nEvaluating subset: ', k)
            if k=='train':
                dl.dataset.mode = 'save_feat'
                dl = data.DataLoader(dl.dataset, shuffle=False, batch_size=dl.batch_size, num_workers=dl.num_workers, pin_memory=dl.pin_memory)
            # initialize records
            records = [record for record in dl.dataset.filenames]
            feature_dict = {r: {'feat': [], 'label': [], 'label_cond': [], 'attrs': [], 'age_p': [], 'pdf_shape': [], 'alpha': []} for r in records}

            # Compute features for each batch
            bar_feat = tqdm(dl, total=len(dl))
            with torch.no_grad():
                for batch in bar_feat:
                    out = self.predict_step(batch)
                    attrs = batch['all_attrs'].copy()

                    # Collect features
                    for i in range(len(out['fids'])):

                        label = out['label'][i].cpu().numpy()
                        label_cond = out['label_cond'][i].cpu().numpy()
                        for key_a, v in batch['all_attrs'].items():
                            attrs[key_a] = v[i].cpu().numpy()
                        feat = out['feat'][i].cpu().numpy()
                        alpha = out['alpha'][i].cpu().numpy().reshape(-1)
                        age_p = out['pred'][i].cpu().numpy()
                        pdf_shape = out['pdf_shape'][i].cpu().numpy()

                        if feature_dict[out['fids'][i]]['label'] == []:
                            feature_dict[out['fids'][i]]['label'] = label
                            feature_dict[out['fids'][i]]['label_cond'] = label_cond
                            feature_dict[out['fids'][i]]['attrs'] = attrs
                            
                        feature_dict[out['fids'][i]]['feat'].append(feat)
                        feature_dict[out['fids'][i]]['alpha'].append(alpha)
                        feature_dict[out['fids'][i]]['age_p'].append(age_p)
                        feature_dict[out['fids'][i]]['pdf_shape'].append(pdf_shape)

            # Save feature dict as h5
            for record, v in feature_dict.items():
                output_filename = os.path.join(self.config.F_train_dir, record)
                with h5py.File(output_filename, "w") as f:
                    # Add datasets
                    f.create_dataset("PSG", data=np.vstack(feature_dict[record]['feat']), dtype='f4')
                    f.create_dataset("alpha", data=np.vstack(feature_dict[record]['alpha']), dtype='f4')
                    f.create_dataset("age_p", data=np.stack(feature_dict[record]['age_p']), dtype='f4')
                    f.create_dataset("pdf_shape", data=np.stack(feature_dict[record]['pdf_shape']), dtype='f4')
                    # Attributes
                    for key_a, v in feature_dict[record]['attrs'].items():
                        f.attrs[key_a] = v

    def log_metrics_to_tensorboard(self, loss, name):
        """logs losses to tensorboard log

        Args:
            loss (list or dict or float): loss function object to write
            name (str): network name
        """
        if self.writer is not None:
            if isinstance(loss, list):
                self.writer.add_scalar(name + '/loss',
                                np.mean([bl['loss'].item() for bl in loss]),
                                self.iter)
                self.writer.add_scalar(name + '/age_l1_loss',
                                np.mean([bl['age_l1_loss'].item() for bl in loss]),
                                self.iter)
            elif isinstance(loss, dict):
                self.writer.add_scalar(name + '/loss',
                                np.array(loss['loss'].item()),
                                self.iter)
                self.writer.add_scalar(name + '/age_l1_loss',
                                np.array(loss['age_l1_loss'].item()),
                                self.iter)
            else:
                self.writer.add_scalar(name + '/loss',
                                np.array(loss.item()),
                                self.iter)
        return

    def activation_maximization(self, save_path = None, n_iter = 1e4, in_size = [1, 12, 128*5*60], lr = 1e-5, l2 = 1e-4):
        """Activation maximization of input polysomnography epoch to maximize age

        Args:
            save_path (str, optional): Path to save AM. Defaults to None.
            n_iter (int, optional): Number to optimization iterations. Defaults to 1e4.
            in_size (list, optional): List of integer specifying the polysomnography input size. Defaults to [1, 12, 128*5*60].
            lr ([type], optional): learning rate for optimization. Defaults to 1e-5.
            l2 ([type], optional): l2 regularization for input (otherwise it explodes). Defaults to 1e-4.
        """
        if self.resume:
            self.restore_checkpoint()


        self.best_network.eval()
        self.best_network.return_only_pred = True

        self.am_model = am_model(self.best_network, in_size).to(self.device)

        self.am_optimizer = torch.optim.Adam(
            [{'params': [p for name, p in self.am_model.named_parameters() if 'am_data' in name], 'weight_decay': l2},
            {'params': [p for name, p in self.am_model.named_parameters() if 'am_data' not in name], 'weight_decay':0}],
            lr=lr)

        self.am_model.train()
        for i in range(int(n_iter)):
            self.am_optimizer.zero_grad()
            output = self.am_model()
            loss = - output.squeeze()
            loss.backward()
            self.am_optimizer.step()
            if i % (n_iter // 100) == 0:
                print('loss: ', loss.item())

        am_data = self.am_model.am_data.detach().cpu().numpy()
        output_filename = os.path.join(save_path, 'am_5.hdf5')
        with h5py.File(output_filename, "w") as f:
            # Save PSG
            f.create_dataset("am_data", data = am_data, dtype='f4')
        return

    def interpret_model(self, test_dataloader, save_path = None, atr_method = 'int_grad'):
        """Interpretation module with sample relevance attribution

        Args:
            test_dataloader (PSG_pretrain_Dataset): A dataloader for interpretation
            save_path (str, optional): Path to save relevance attribution. Defaults to None.
            atr_method (str, optional): Relevance attribution method. Select one of 
            ['int_grad', 'grad_shap', 'deep_lift', 'deep_lift_shap', 'int_smooth_grad', 'inputXgradient', 'occlusion']. Defaults to 'int_grad'.
        """
        if self.resume:
            self.restore_checkpoint()

        # initialize records
        #records = [record for record in test_dataloader.dataset.filenames]
        #predictions = {r: {'interpretation': [],'delta': []} for r in records}

        print(f'\nComputing model interpretation')
        # Model train to enable gradient computation
        self.best_network.eval()
        self.best_network.LSTM.training = True
        self.best_network.return_only_pred = True
        #self.best_network.dropout.p = 0.0

        if atr_method == 'int_grad':
            i_model = IntegratedGradients(self.best_network)
        elif atr_method == 'grad_shap':
            i_model = GradientShap(self.best_network)
        elif atr_method == 'deep_lift':
            i_model = DeepLift(self.best_network)
        elif atr_method == 'deep_lift_shap':
            i_model = DeepLiftShap(self.best_network)
        elif atr_method == 'int_smooth_grad':
            i_model = IntegratedGradients(self.best_network)
            i_model = NoiseTunnel(i_model)
        elif atr_method == 'inputXgradient':
            i_model = InputXGradient(self.best_network)
        #elif atr_method == 'occlusion':


        bar_test = tqdm(test_dataloader, total=len(test_dataloader))
        current_subj = ''
        for batch in bar_test:

            fids = batch['fid']
            pos = batch['position']
            x = batch['data'].to(self.device)

            # Run interpretation
            if atr_method == 'int_grad':
                baseline = torch.zeros_like(x)
                attributions, delta = i_model.attribute(x, baseline, target=0, n_steps = 10, return_convergence_delta=True)
            elif atr_method == 'grad_shap':
                baseline_dist = torch.randn(x.size(0)*5, x.size(1), x.size(2)).to(self.device)*0.001
                attributions, delta = i_model.attribute(x, stdevs=0.09, n_samples=4, baselines=baseline_dist, target=0, return_convergence_delta=True)
                delta = torch.mean(delta.reshape(x.shape[0], -1), dim=1)
            elif atr_method == 'deep_lift':
                baseline = torch.zeros_like(x)
                attributions, delta = i_model.attribute(x, baseline, target=0, return_convergence_delta=True)
            elif atr_method == 'deep_lift_shap':
                baseline_dist = torch.randn(x.size(0)*5, x.size(1), x.size(2)).to(self.device)*0.001
                attributions, delta = i_model.attribute(x, baseline_dist, target=0, return_convergence_delta=True)
                delta = torch.mean(delta.reshape(x.shape[0], -1), dim=1)
            elif atr_method == 'int_smooth_grad':
                baseline = torch.zeros_like(x)
                attributions, delta = i_model.attribute(x, baselines=baseline, nt_type='smoothgrad', stdevs=0.02, target=0, n_samples=5, n_steps = 5, return_convergence_delta=True)
            elif atr_method == 'inputXgradient':
                attributions = i_model.attribute(x, target=0)
                delta = torch.zeros_like(x)
            elif atr_method == 'occlusion':
                attributions, delta = self.occlusion_attribution(x)

            # Assign to subjects
            for record, b_pos, seq_interpretation, seq_delta in zip(fids, pos, attributions, delta):
                if record == current_subj:
                    interpretation = np.concatenate((interpretation, seq_interpretation.cpu().detach().numpy()), 1)
                    err_delta = np.concatenate((err_delta, np.expand_dims(seq_delta.cpu().detach().numpy(), 0)), 0)
                    rec_pos.append(b_pos[0])
                else:
                    if current_subj != '' and save_path is not None:
                        # Save interpretation as h5 files
                        output_filename = os.path.join(save_path, current_subj)
                        with h5py.File(output_filename, "w") as f:
                            # Save PSG
                            f.create_dataset("Interpretation", data = interpretation, dtype='f4')
                            f.create_dataset("Delta", data = err_delta, dtype='f4')
                            f.create_dataset("Position", data = np.array(rec_pos), dtype='i4')
                    # Create new array for new recording
                    current_subj = record
                    interpretation = seq_interpretation.cpu().detach().numpy()
                    err_delta = np.expand_dims(seq_delta.cpu().detach().numpy(), 0)
                    rec_pos = [b_pos[0]]

        # Save interpretation as h5 files
        output_filename = os.path.join(save_path, current_subj)
        with h5py.File(output_filename, "w") as f:
            # Save PSG
            f.create_dataset("Interpretation", data = interpretation, dtype='f4')
            f.create_dataset("Delta", data = err_delta, dtype='f4')
            f.create_dataset("Position", data = np.array(rec_pos), dtype='i4')

        return

    def occlusion_attribution(self, x):
        """Occlusion analysis

        Args:
            x (Tensor): Input polysomnography epoch

        Returns:
            attribution (Tensor): Relevance attribution scores
            delta (Tensor): Error estimate of attribution scores (not estimated in this function)
        """
        
        occlude_channel_sets = [[0, 1],[2, 3],[4],[5],[6],[7,8,9,10],[11]]
        occlude_ref_style = ['same','same','same','same','same','same']
        occlude_window = 5*128

        y = self.best_network(x)

        y_occ_all = torch.zeros_like(x)
        
        for occ_set in occlude_channel_sets:
            for occ_start in range(0, 5*128*60, occlude_window):
                x_occ = x.detach().clone()
                x_occ[:, occ_set, occ_start:(occ_start + occlude_window)] = 0
                y_occ = self.best_network(x_occ)
                y_occ_all[:, occ_set, occ_start:(occ_start + occlude_window)] = y_occ.detach().clone().unsqueeze(2).repeat(1,len(occ_set), occlude_window)

        attribution = y_occ_all - y.detach().clone().unsqueeze(2).repeat(1, len(occ_set), 128*60*5)
        return attribution, torch.zeros_like(x)
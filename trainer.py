import os
import h5py

import numpy as np
import pandas as pd

import torch
from torch.utils import data

from tqdm import tqdm

from base.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, network, loss_fn, config, writer,
                 device='cpu',
                 num_epochs=100,
                 patience=None,
                 resume=None,
                 scheduler=None):
        """Class to train an age estimation network

        Args:
            network (M_FEAT2LABEL): An network for age estimation in polysomnographs
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
            self.iter = self.current_epoch * self.len_epoch // self.config.batch_size

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
                self.log_metrics_to_tensorboard(loss_out, 'Training')
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
            self.log_metrics_to_tensorboard(batch_loss_outs, 'Validation')

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
        x, y, z = batch['data'].to(self.device), batch['label'].to(self.device), batch['label_cond'].to(self.device)

        # Reset gradients
        self.optimizer.zero_grad()

       # Run forward pass
        if self.network.return_only_pred:
            y_p = self.network(x, z)
        else:
            out = self.network(x, z)
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
        x, y, z = batch['data'].to(self.device), batch['label'].to(self.device), batch['label_cond'].to(self.device)

        # Run forward pass
        if self.network.return_only_pred:
            y_p = self.network(x, z)
        else:
            out = self.network(x, z)
            y_p = out['pred']
            if self.nll_loss:
                pdf_shape = out['pdf_shape']

        # Calculate training loss
        if self.nll_loss:
            loss_out = self.loss_fn(y_p, y, pdf_shape)
        else:
            loss_out = self.loss_fn(y_p, y)

        return loss_out

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
            y_p = self.best_network(x, z)
            out = {'pred': y_p}
        else:
            out = self.best_network(x, z)
        
        # Add record and label info
        out['fids'] = record
        out['position'] = position
        out['label'] = y
        out['label_cond'] = z

        return out

    def evaluate_performance(self, test_dataloader, len_epoch = 0):
        """Test performance on test set

        Args:
            test_dataloader (PSG_feature_Dataset): A test dataset
            len_epoch (int, optional): Training epoch length, which is used to log performance. Defaults to 0.

        Returns:
            predictions: age predictions per epoch and attention weights
            metrics: a dict containing all record names, ages, average age predictions, and losses
        """
        self.len_epoch = len_epoch

        if self.resume:
            self.restore_checkpoint()
            self.iter = self.last_update * self.len_epoch // self.config.batch_size

        # initialize records
        records = [record for record in test_dataloader.dataset.filenames]
        predictions = {r: {'age_p': [], 'pdf_shape': [], 'label': [], 'alpha': []} for r in records}
        metrics = {'record': records,
                   'age': [],
                   'age_p': [],
                   'pdf_shape': [],
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
                y = batch['label'].numpy()
                x = batch['data'].to(self.device)
                z = batch['label_cond'].to(self.device)

                # Run forward pass and get predictions
                if self.best_network.return_only_pred:
                    y_p = self.best_network(x, z)
                else:
                    out = self.best_network(x, z)
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
                for record, pred, r_pdf_shape, labels, alpha_w in zip(fids, y_p, pdf_shape, y, alpha):
                    # Predictions
                    predictions[record]['age_p'].append(pred.cpu().numpy().tolist())
                    predictions[record]['pdf_shape'].append(r_pdf_shape.cpu().numpy().tolist())
                    predictions[record]['label'].append(labels[0])
                    predictions[record]['alpha'].append(alpha_w.cpu().numpy().reshape(-1).tolist())

        # Log to tensorboard
        self.log_metrics_to_tensorboard(batch_loss_outs, 'Test')

        # Calculate metrics
        for record in metrics['record']:

            y = predictions[record]['label']
            y_p = predictions[record]['age_p']
            pdf_shape = predictions[record]['pdf_shape']

            # Log labels
            metrics['age'].append(y[0])
            
            # Log prediction
            metrics['age_p'].append(y_p[0])

            # PDF shape
            metrics['pdf_shape'].append(pdf_shape[0])

            # Log loss and accuracy
            if self.nll_loss:
                loss_out = self.loss_fn(torch.Tensor(y_p).to(self.device), torch.Tensor([[y[0]]]).to(self.device), torch.Tensor(pdf_shape).to(self.device))
            else:
                loss_out = self.loss_fn(torch.Tensor(y_p).to(self.device), torch.Tensor([[y[0]]]).to(self.device))
                
            metrics['loss'].append(loss_out['loss'].item())
            metrics['age_l1_loss'].append(loss_out['age_l1_loss'].item())

        # log covariance of error and true age
        test_cov = np.cov(np.array(metrics['age_p'])-np.array(metrics['age']), np.array(metrics['age']))
        self.writer.add_scalar('Test/covariance', test_cov[1, 0], self.iter)

        return pd.DataFrame.from_dict(metrics).set_index('record'), predictions

    def log_metrics_to_tensorboard(self, loss, name):
        """logs losses to tensorboard log

        Args:
            loss (list or dict or float): loss function object to write
            name (str): network name
        """
        if isinstance(loss, list):
            self.writer.add_scalar(name + '/loss',
                              np.mean([bl['loss'].item() for bl in loss]),
                              self.iter)
            self.writer.add_scalar(name + '/age_l1_loss',
                              np.mean([bl['age_l1_loss'].item() for bl in loss]),
                              self.iter)
            self.writer.add_scalar(name + '/loss_cov',
                              np.mean([bl['loss_cov'].item() for bl in loss]),
                              self.iter)
        elif isinstance(loss, dict):
            self.writer.add_scalar(name + '/loss',
                              np.array(loss['loss'].item()),
                              self.iter)
            self.writer.add_scalar(name + '/age_l1_loss',
                              np.array(loss['age_l1_loss'].item()),
                              self.iter)
            self.writer.add_scalar(name + '/loss_cov',
                              np.array(loss['loss_cov'].item()),
                              self.iter)
        else:
            self.writer.add_scalar(name + '/loss',
                              np.array(loss.item()),
                              self.iter)
        return



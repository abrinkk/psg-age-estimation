import os
import h5py

import numpy as np
import pandas as pd

import torch

from tqdm import tqdm

from base.base_trainer import BaseTrainer

class PreTrainer(BaseTrainer):

    def __init__(self, network, loss_fn, config, writer,
                 device='cpu',
                 num_epochs=100,
                 patience=None,
                 resume=None,
                 scheduler=None):
        super().__init__(network, loss_fn, config,
                         device=device,
                         num_epochs=num_epochs,
                         patience=patience,
                         resume=resume,
                         scheduler=scheduler)
        self.writer = writer
        self.iter = 0
        self.len_epoch = 0

    def train_and_validate(self, train_dataloader, eval_dataloader):

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
        x, y = batch['data'].to(self.device), batch['label'].to(self.device)

        # Reset gradients
        self.optimizer.zero_grad()

        # Run forward pass
        if self.network.return_only_pred:
            y_p = self.network(x)
        else:
            out = self.network(x)
            y_p = out['pred']

        # Calculate training loss
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
        x, y = batch['data'].to(self.device), batch['label'].to(self.device)

        # Run forward pass
        if self.network.return_only_pred:
            y_p = self.network(x)
        else:
            out = self.network(x)
            y_p = out['pred']

        # Calculate loss
        loss_out = self.loss_fn(y_p, y)

        return loss_out

    def predict_step(self, batch):
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

    def evaluate_performance(self, test_dataloader):

        # initialize records
        records = [record for record in test_dataloader.dataset.filenames]
        predictions = {r: {'age_p': [], 'bmi_p': [], 'sex_p': [], 'label': [], 'alpha': []} for r in records}
        metrics = {'record': records,
                   'age': [],
                   'bmi': [],
                   'sex': [],
                   'age_p': [],
                   'bmi_p': [],
                   'sex_p': [],
                   'loss': [],
                   'age_l1_loss': [],
                   'bmi_l1_loss': [],
                   'sex_ce_loss': [],
                   'sex_acc': []}

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

                # Run forward pass and get predictions
                if self.best_network.return_only_pred:
                    y_p = self.best_network(x)
                else:
                    out = self.best_network(x)
                    y_p = out['pred']
                    alpha = out['alpha']
                
                # Compute loss
                loss_out = self.loss_fn(y_p, y.to(self.device))
                if isinstance(loss_out, dict):
                    loss = loss_out['loss'].item()
                else:
                    loss = loss_out.item()
                batch_losses.append(loss)
                batch_loss_outs.append(loss_out)
                
                # Assign to subjects
                for record, pred, labels, alpha_w in zip(fids, y_p, y, alpha):
                    # Predictions
                    predictions[record]['age_p'].append(pred[0].cpu().numpy())
                    predictions[record]['bmi_p'].append(pred[1].cpu().numpy())
                    predictions[record]['sex_p'].append(pred[2].cpu().numpy())
                    predictions[record]['labels'].append(labels)
                    predictions[record]['alpha'].append(alpha_w)

        # Log to tensorboard
        self.log_metrics_to_tensorboard(batch_loss_outs, 'PreTest')

        # Calculate metrics
        for record in metrics['record']:

            y = predictions[record]['labels']
            y_p = [[np.mean(predictions[record]['age_p']),
                    np.mean(predictions[record]['bmi_p']),
                    np.mean(predictions[record]['sex_p']),
                    1.0 - np.mean(predictions[record]['sex_p'])]]

            # Log labels
            metrics['age'].append(y[0])
            metrics['bmi'].append(y[1])
            metrics['sex'].append(y[2])
            
            # Log averaged predictions
            metrics['age_p'].append(y_p[0][0])
            metrics['bmi_p'].append(y_p[0][1])
            metrics['sex_p'].append(np.array(y_p[0][2:]).argmax())

            # Log loss and accuracy
            loss_out = self.loss_fn(torch.Tensor(y_p).to(self.device),torch.Tensor(y).to(self.device))
            metrics['loss'].append(loss_out['loss'].item())
            metrics['age_l1_loss'].append(loss_out['age_l1_loss'].item())
            metrics['bmi_l1_loss'].append(loss_out['bmi_l1_loss'].item())
            metrics['sex_ce_loss'].append(loss_out['sex_ce_loss'].item())
            metrics['sex_acc'].append(loss_out['sex_acc'].item())

        return pd.DataFrame.from_dict(metrics).set_index('record'), predictions

    def save_features(self, dataloaders):

        print(f'\nEvaluating features')
        self.best_network.eval()

        # Iterate dataloaders
        for k, dl in dataloaders.items():

            print(f'\nEvaluating subset: ', k)
            dl.Shuffle=False

            # initialize records
            records = [record for record in dl.dataset.filenames]
            feature_dict = {r: {'feat': [], 'label': [], 'label_cond': [], 'age_p': [], 'bmi_p': [], 'sex_p': [], 'alpha': []} for r in records}

            # Compute features for each batch
            bar_feat = tqdm(dl, total=len(dl))
            with torch.no_grad():
                for batch in bar_feat:
                    out = self.predict_step(batch)

                    # Collect features
                    for i in range(out['fids']):

                        age_p = out['pred'][i][0]
                        bmi_p = out['pred'][i][1]
                        sex_p = np.array(out['pred'][i][2:]).argmax()

                        feature_dict[out['fids'][i]]['label']=out['label'][i]
                        feature_dict[out['fids'][i]]['label_cond']=out['label_cond'][i]
                        feature_dict[out['fids'][i]]['feat'].append(out['feat'][i])
                        feature_dict[out['fids'][i]]['alpha'].append(out['alpha'][i])
                        feature_dict[out['fids'][i]]['age_p'].append(age_p)
                        feature_dict[out['fids'][i]]['bmi_p'].append(bmi_p)
                        feature_dict[out['fids'][i]]['sex_p'].append(sex_p)

            # Save feature dict as h5
            for record, v in feature_dict.items():
                output_filename = os.path.join(self.config.data_dir, k + '_F', record)
                z = feature_dict[record]['label_cond']
                with h5py.File(output_filename, "w") as f:
                    # Add datasets
                    f.create_dataset("PSG", data=feature_dict[record]['feat'], dtype='f4')
                    f.create_dataset("alpha", data=feature_dict[record]['alpha'], dtype='f4')
                    f.create_dataset("age_p", data=feature_dict[record]['age_p'], dtype='f4')
                    f.create_dataset("bmi_p", data=feature_dict[record]['bmi_p'], dtype='f4')
                    f.create_dataset("sex_p", data=feature_dict[record]['sex_p'], dtype='f4')
                    # Labels
                    for l, l_name in zip(feature_dict[record]['label'], self.config.pre_label):
                        f.attrs[l_name]=l
                    # Conditioning labels
                    cond_idx_start = 0
                    for idx, i in enumerate(self.config.label_cond):
                        cond_idx_end = cond_idx_start + self.config.label_cond_size[idx]
                        f.attrs[i] = z[cond_idx_start:cond_idx_end]
                        cond_idx_start = cond_idx_end

    def log_metrics_to_tensorboard(self, loss, name):
        if isinstance(loss, list):
            self.writer.add_scalar(name + '/loss',
                              np.mean([bl['loss'].item() for bl in loss]),
                              self.iter)
            self.writer.add_scalar(name + '/age_l1_loss',
                              np.mean([bl['age_l1_loss'].item() for bl in loss]),
                              self.iter)
            self.writer.add_scalar(name + '/bmi_l1_loss',
                              np.mean([bl['bmi_l1_loss'].item() for bl in loss]),
                              self.iter)
            self.writer.add_scalar(name + '/sex_ce_loss',
                              np.mean([bl['sex_ce_loss'].item() for bl in loss]),
                              self.iter)
            self.writer.add_scalar(name + '/sex_acc',
                              np.mean([bl['sex_acc'].item() for bl in loss]),
                              self.iter)
        elif isinstance(loss, dict):
            self.writer.add_scalar(name + '/loss',
                              np.array(loss['loss'].item()),
                              self.iter)
            self.writer.add_scalar(name + '/age_l1_loss',
                              np.array(loss['age_l1_loss'].item()),
                              self.iter)
            self.writer.add_scalar(name + '/bmi_l1_loss',
                              np.array(loss['bmi_l1_loss'].item()),
                              self.iter)
            self.writer.add_scalar(name + '/sex_ce_loss',
                              np.array(loss['sex_ce_loss'].item()),
                              self.iter)
            self.writer.add_scalar(name + '/sex_acc',
                              np.array(loss['sex_acc'].item()),
                              self.iter)
        else:
            self.writer.add_scalar(name + '/loss',
                              np.array(loss.item()),
                              self.iter)
        return



import ax
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from trainer import Trainer

np.random.seed(0)
torch.manual_seed(0)

def train_eval(args, config, network_fn, loss_fn, dataloaders, device, write_path, params):

    # Set up config to match hyper-parameters
    hyper_param_string = 'dol_{:.3f}_lr_{:.5f}_l2_{:.7f}_size_{:.0f}_lstmn_{:.0f}'.format(params['do'], params['lr'], params['l2'], params['size'], params['lstmn'])
    config.do_l = params['do']
    config.lr = params['lr']
    config.l2 = params['l2']
    config.net_size_scale = params['size']
    config.lstm_n = params['lstmn']
    config.save_dir = config.model_L_BO_path

    # Set up new writer
    writer = SummaryWriter(write_path + '_' + hyper_param_string)

    # Initiate new model
    network = network_fn(config).to(device)

    # Initiate new trainer
    trainer = Trainer(network, loss_fn, config, writer,
                    device=device,
                    num_epochs=config.max_epochs,
                    patience=config.patience,
                    resume=args.train_resume)
    
    # Train and evaluate model with trainer
    trainer.train_and_validate(dataloaders['train'], dataloaders['val'])
    perf = trainer.best_loss

    print('Loss: {:.6f}. Dropout: {:.3f}. Learning rate: {:.5f}. L2 weight decay: {:.7f}. Size: {:.0f}. LSTM n: {:.0f}.'.format(perf, params['do'], params['lr'], params['l2'], params['size'], params['lstmn']))

    return perf

def bayesian_opt_eval(args, config, network_fn, loss_fn, dataloaders, device, write_path):

    exp = ax.SimpleExperiment(
        name='age_label_experiment',
        search_space=ax.SearchSpace(
            parameters=[
                ax.RangeParameter(name='do', lower=0.0, upper=0.99, parameter_type=ax.ParameterType.FLOAT),
                ax.RangeParameter(name='lr', lower=10**(-6), upper=10**(-2), parameter_type=ax.ParameterType.FLOAT, log_scale=True),
                ax.RangeParameter(name='l2', lower=10**(-8), upper=10**(-3), parameter_type=ax.ParameterType.FLOAT, log_scale=True),
                ax.RangeParameter(name='size', lower=1, upper=10, parameter_type=ax.ParameterType.INT),
                ax.RangeParameter(name='lstmn', lower=1, upper=3, parameter_type=ax.ParameterType.INT),
            ]
        ),
        evaluation_function=lambda p: train_eval(args, config, network_fn, loss_fn, dataloaders, device, write_path, p),
        minimize=True
    )

    sobol = ax.Models.SOBOL(exp.search_space)
    for i in range(10):
        exp.new_trial(generator_run=sobol.gen(1))

    best_arm = None
    for i in range(20):
        gpei = ax.Models.GPEI(experiment=exp, data=exp.eval())
        generator_run = gpei.gen(1)
        best_arm, _ = generator_run.best_arm_predictions
        exp.new_trial(generator_run=generator_run)

    best_parameters = best_arm.parameters
    ax.save(exp, config.BO_expe_path + '.json')
    return best_parameters

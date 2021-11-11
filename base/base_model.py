import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError

    def summary(self, input_size, device, batch_size=-1):

        def register_hook(module):

            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[(-1)].split("'")[0]
                module_idx = len(summary)
                m_key = '%s-%i' % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = batch_size
                if isinstance(output, (list, tuple)):
                    if class_name in ('GRU', 'LSTM', 'RNN'):
                        summary[m_key]['output_shape'] = [
                            batch_size] + list(output[0].size())[1:]
                    else:
                        summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
                else:
                    summary[m_key]['output_shape'] = list(output.size())
                    summary[m_key]['output_shape'][0] = batch_size
                summary[m_key]['trainable'] = any([p.requires_grad for p in module.parameters()])
                params = np.sum([np.prod(list(p.size())) for p in module.parameters() if p.requires_grad])
                summary[m_key]['nb_params'] = int(params)

            if not isinstance(module, nn.Sequential):
                if not isinstance(module, nn.ModuleList):
                    pass
            if not module == self:
                hooks.append(module.register_forward_hook(hook))

        assert device.type in ('cuda', 'cpu'), "Input device is not valid, please specify 'cuda' or 'cpu'"
        if device.type == 'cuda':
            if torch.cuda.is_available():
                dtype = torch.cuda.FloatTensor
            else:
                dtype = torch.FloatTensor
            if isinstance(input_size, tuple):
                input_size = [
                    input_size]
            x = [(torch.rand)(*(2, ), *in_size).type(dtype) for in_size in input_size]
            summary = OrderedDict()
            hooks = []
            self.apply(register_hook)
            self(*x)
            for h in hooks:
                h.remove()

            print('----------------------------------------------------------------')
            line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
            print(line_new)
            print('================================================================')
            total_params = 0
            total_output = 0
            trainable_params = 0
            for layer in summary:
                line_new = '{:>20}  {:>25} {:>15}'.format(layer, str(
                    summary[layer]['output_shape']), '{0:,}'.format(summary[layer]['nb_params']))
                total_params += summary[layer]['nb_params']
                if any(isinstance(el, list) for el in summary[layer]['output_shape']):
                    for list_out in summary[layer]['output_shape']:
                        total_output += np.prod(list_out,dtype = np.int64)
                else:
                    total_output += np.prod(summary[layer]['output_shape'],dtype = np.int64)
                if 'trainable' in summary[layer]:
                    if summary[layer]['trainable'] == True:
                        trainable_params += summary[layer]['nb_params']
                    print(line_new)
            total_input_size = abs(np.sum([np.prod(x) for x in input_size]) * batch_size * 4.0 / 1073741824.0)
            total_output_size = abs(2.0 * total_output * 4.0 / 1073741824.0)
            total_params_size = abs(total_params * 4.0 / 1073741824.0)
            total_size = total_params_size + total_output_size + total_input_size
            print('================================================================')
            print('Total params: {0:,}'.format(total_params))
            print('Trainable params: {0:,}'.format(trainable_params))
            print('Non-trainable params: {0:,}'.format(total_params - trainable_params))
            print('----------------------------------------------------------------')
            print('Input size (GB): %0.2f' % total_input_size)
            print('Forward/backward pass size (GB): %0.2f' % total_output_size)
            print('Params size (GB): %0.2f' % total_params_size)
            print('Estimated Total Size (GB): %0.2f' % total_size)
            print('----------------------------------------------------------------')

    def debug_model(self, input_size, device, cond_size = False):
        if cond_size or cond_size is 0:
            self.summary([input_size[1:], (cond_size,)], device, input_size[0])
            z = torch.rand((input_size[0],cond_size)).to(device)
        else:
            self.summary(input_size[1:], device, input_size[0])
        X = torch.rand(input_size).to(device)
        print('Input size: ', X.size)
        time_start = time.time()
        if cond_size or cond_size is 0:
            out = self(X, z)
        else:
            out = self(X)
        print('Batch time: {:.3f}'.format(time.time() - time_start))
        for k, v in out.items():
            print('Key: ', k)
            print('Output size: ', v.size())

import sys
from collections import OrderedDict

import torch
from torch import nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, cuda=False):
        super(Model, self).__init__()
        from models.step1 import step1
        from models.step2 import step2
        self.step1 = step1
        self.step1.load_state_dict(torch.load('models/step1.pth'))
        self.step2 = step2
        self.step2.load_state_dict(torch.load('models/step2.pth'))
        self.silhouette_threshold = 30
        if cuda is not False:
            self.cuda(cuda if cuda is not True else None)

    def forward(self, input):
        step1_out = self.step1(input)
        masked_normal_depth = self.mask(step1_out)
        step2_out = self.step2(masked_normal_depth)
        assert list(step2_out.size())[-3:] == [128, 128, 128]
        return step2_out

    def mask(self, input):
        normal = input[0]
        depth = input[1]
        silhouette = input[2][:, 0, :, :]
        silhouette_mask = torch.le(silhouette, self.silhouette_threshold)

        # TODO: solve this within PyTorch. Work around in-place limitation
        masked_normal = normal.data.numpy()
        masked_depth = depth.data.numpy()
        silhouette_mask = silhouette_mask.data.numpy().astype(bool)
        for i in range(3):
            masked_normal[:, i, :, :][silhouette_mask] = 100
        masked_depth[:, 0, :, :][silhouette_mask] = 0
        return torch.cat((Variable(torch.FloatTensor(masked_normal)), Variable(torch.FloatTensor(masked_depth))), 1)


def summary(model, x, file=sys.stdout):
    """
    adapted from https://github.com/pytorch/pytorch/pull/3043
    Use input x directly instead of creating from input_size
    """

    def register_hook(module):
        def hook(module, input, output):
            if module._modules:  # only want base layers
                return
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = None
            if output.__class__.__name__ == 'tuple':
                summary[m_key]['output_shape'] = list(output[0].size())
            else:
                summary[m_key]['output_shape'] = list(output.size())
            summary[m_key]['output_shape'][0] = None

            params = 0
            # iterate through parameters and count num params
            for name, p in module._parameters.items():
                params += torch.numel(p.data) if p is not None else 0
                summary[m_key]['trainable'] = p.requires_grad if p is not None else False

            summary[m_key]['nb_params'] = params

        if not isinstance(module, torch.nn.Sequential) and \
                not isinstance(module, torch.nn.ModuleList) and \
                not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    # print out neatly
    def get_names(module, name, acc):
        if not module._modules:
            acc.append(name)
        else:
            for key in module._modules.keys():
                p_name = key if name == "" else name + "." + key
                get_names(module._modules[key], p_name, acc)

    names = []
    get_names(model, "", names)

    col_width = 25  # should be >= 12
    summary_width = 61

    def crop(s):
        return s[:col_width] if len(s) > col_width else s

    print('_' * summary_width, file=file)
    print('{0: <{3}} {1: <{3}} {2: <{3}}'.format(
        'Layer (type)', 'Output Shape', 'Param #', col_width), file=file)
    print('=' * summary_width, file=file)
    total_params = 0
    trainable_params = 0
    for (i, l_type), l_name in zip(enumerate(summary), names):
        d = summary[l_type]
        total_params += d['nb_params']
        if 'trainable' in d and d['trainable']:
            trainable_params += d['nb_params']
        print('{0: <{3}} {1: <{3}} {2: <{3}}'.format(
            crop(l_name + ' (' + l_type[:-2] + ')'), crop(str(d['output_shape'])),
            crop(str(d['nb_params'])), col_width), file=file)
        if i < len(summary) - 1:
            print('_' * summary_width, file=file)
    print('=' * summary_width, file=file)
    print('Total params: ' + str(total_params), file=file)
    print('Trainable params: ' + str(trainable_params), file=file)
    print('Non-trainable params: ' + str((total_params - trainable_params)), file=file)
    print('_' * summary_width, file=file)

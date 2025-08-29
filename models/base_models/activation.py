'''
Function:
    Implementation of ActivationBuilder and BuildActivation
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import copy
import collections


'''BaseModuleBuilder'''
class BaseModuleBuilder():
    REGISTERED_MODULES = collections.OrderedDict()
    def __init__(self, requires_register_modules=None, requires_renew_modules=None):
        if requires_register_modules is not None and isinstance(requires_register_modules, (dict, collections.OrderedDict)):
            for name, module in requires_register_modules.items():
                self.register(name, module)
        if requires_renew_modules is not None and isinstance(requires_renew_modules, (dict, collections.OrderedDict)):
            for name, module in requires_renew_modules.items():
                self.renew(name, module)
        self.validate()
    '''build'''
    def build(self, module_cfg):
        module_cfg = copy.deepcopy(module_cfg)
        module_type = module_cfg.pop('type')
        module = self.REGISTERED_MODULES[module_type](**module_cfg)
        return module
    '''register'''
    def register(self, name, module):
        assert callable(module)
        assert name not in self.REGISTERED_MODULES
        self.REGISTERED_MODULES[name] = module
    '''renew'''
    def renew(self, name, module):
        assert callable(module)
        assert name in self.REGISTERED_MODULES
        self.REGISTERED_MODULES[name] = module
    '''validate'''
    def validate(self):
        for _, module in self.REGISTERED_MODULES.items():
            assert callable(module)
    '''delete'''
    def delete(self, name):
        assert name in self.REGISTERED_MODULES
        del self.REGISTERED_MODULES[name]
    '''pop'''
    def pop(self, name):
        assert name in self.REGISTERED_MODULES
        module = self.REGISTERED_MODULES.pop(name)
        return module
    '''get'''
    def get(self, name):
        assert name in self.REGISTERED_MODULES
        module = self.REGISTERED_MODULES.get(name)
        return module
    '''items'''
    def items(self):
        return self.REGISTERED_MODULES.items()
    '''clear'''
    def clear(self):
        return self.REGISTERED_MODULES.clear()
    '''values'''
    def values(self):
        return self.REGISTERED_MODULES.values()
    '''keys'''
    def keys(self):
        return self.REGISTERED_MODULES.keys()
    '''copy'''
    def copy(self):
        return self.REGISTERED_MODULES.copy()
    '''update'''
    def update(self, requires_update_modules):
        return self.REGISTERED_MODULES.update(requires_update_modules)


'''HardSigmoid'''
class HardSigmoid(nn.Module):
    def __init__(self, bias=1.0, divisor=2.0, min_value=0.0, max_value=1.0):
        super(HardSigmoid, self).__init__()
        assert divisor != 0, 'divisor is not allowed to be equal to zero'
        self.bias = bias
        self.divisor = divisor
        self.min_value = min_value
        self.max_value = max_value
    '''forward'''
    def forward(self, x):
        x = (x + self.bias) / self.divisor
        return x.clamp_(self.min_value, self.max_value)



'''Swish'''
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    '''forward'''
    def forward(self, x):
        return x * torch.sigmoid(x)

'''HardSwish'''
class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.act = nn.ReLU6(inplace)
    '''forward'''
    def forward(self, x):
        return x * self.act(x + 3) / 6

'''ActivationBuilder'''
class ActivationBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'ReLU': nn.ReLU, 'GELU': nn.GELU, 'ReLU6': nn.ReLU6, 'PReLU': nn.PReLU,
        'Sigmoid': nn.Sigmoid, 'HardSwish': HardSwish, 'LeakyReLU': nn.LeakyReLU,
        'HardSigmoid': HardSigmoid, 'Swish': Swish,
    }
    for act_type in ['ELU', 'Hardshrink', 'Hardtanh', 'LogSigmoid', 'RReLU', 'SELU', 'CELU', 'SiLU', 'GLU', 
                     'Mish', 'Softplus', 'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink', 'Threshold']:
        if hasattr(nn, act_type):
            REGISTERED_MODULES[act_type] = getattr(nn, act_type)
    '''build'''
    def build(self, act_cfg):
        if act_cfg is None: return nn.Identity()
        return super().build(act_cfg)


'''BuildActivation'''
BuildActivation = ActivationBuilder().build
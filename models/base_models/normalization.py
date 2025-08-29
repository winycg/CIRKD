'''
Function:
    Implementation of NormalizationBuilder and BuildNormalization
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
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


'''GRN'''
class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(GRN, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    '''forward'''
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x


'''LayerNorm2d'''
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, **kwargs):
        super(LayerNorm2d, self).__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]
    '''forward'''
    def forward(self, x):
        assert x.dim() == 4, f'LayerNorm2d only supports inputs with shape (N, C, H, W), but got tensor with shape {x.shape}'
        x = F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)
        return x



'''NormalizationBuilder'''
class NormalizationBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'LayerNorm': nn.LayerNorm, 'LayerNorm2d': LayerNorm2d, 'GroupNorm': nn.GroupNorm, 'LocalResponseNorm': nn.LocalResponseNorm, 
        'BatchNorm1d': nn.BatchNorm1d, 'BatchNorm2d': nn.BatchNorm2d, 'BatchNorm3d': nn.BatchNorm3d, 'SyncBatchNorm': nn.SyncBatchNorm, 
        'InstanceNorm1d': nn.InstanceNorm1d, 'InstanceNorm2d': nn.InstanceNorm2d, 'InstanceNorm3d': nn.InstanceNorm3d, 'GRN': GRN,
    }
    for norm_type in ['LazyBatchNorm1d', 'LazyBatchNorm2d', 'LazyBatchNorm3d', 'LazyInstanceNorm1d', 'LazyInstanceNorm2d', 'LazyInstanceNorm3d']:
        if hasattr(nn, norm_type):
            REGISTERED_MODULES[norm_type] = getattr(nn, norm_type)
    '''build'''
    def build(self, placeholder, norm_cfg):
        if norm_cfg is None: return nn.Identity()
        norm_cfg = copy.deepcopy(norm_cfg)
        norm_type = norm_cfg.pop('type')
        if norm_type in ['GroupNorm']:
            normalization = self.REGISTERED_MODULES[norm_type](num_channels=placeholder, **norm_cfg)
        else:
            normalization = self.REGISTERED_MODULES[norm_type](placeholder, **norm_cfg)
        return normalization
    '''isnorm'''
    @staticmethod
    def isnorm(module, norm_list=None):
        if norm_list is None:
            norm_list = (
                nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.SyncBatchNorm,
            )
        return isinstance(module, norm_list)


'''BuildNormalization'''
BuildNormalization = NormalizationBuilder().build
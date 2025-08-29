'''
Function:
    Implementation of DropoutBuilder and BuildDropout
Author:
    Zhenchao Jin
'''
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


'''DropPath'''
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob
    '''forward'''
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(self.keep_prob) * random_tensor
        return output

'''DropoutBuilder'''
class DropoutBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'DropPath': DropPath, 'Dropout': nn.Dropout, 'Dropout2d': nn.Dropout2d, 'Dropout3d': nn.Dropout3d,
    }
    '''build'''
    def build(self, dropout_cfg):
        if dropout_cfg is None: return nn.Identity()
        return super().build(dropout_cfg)


'''BuildDropout'''
BuildDropout = DropoutBuilder().build
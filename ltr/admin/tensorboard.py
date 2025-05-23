import os
from collections import OrderedDict
import wandb
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    print('WARNING: You are using tensorboardX instead sis you have a too old pytorch version.')
    from tensorboardX import SummaryWriter


class TensorboardWriter:
    def __init__(self, directory, loader_names):
        self.directory = directory
        self.writer = OrderedDict({name: SummaryWriter(os.path.join(self.directory, name)) for name in loader_names})

    def write_info(self, module_name, script_name, description):
        tb_info_writer = SummaryWriter(os.path.join(self.directory, 'info'))
        tb_info_writer.add_text('Modulet_name', module_name)
        tb_info_writer.add_text('Script_name', script_name)
        tb_info_writer.add_text('Description', description)
        tb_info_writer.close()

    def write_epoch(self, stats: OrderedDict, epoch: int, ind=-1):
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue
            for var_name, val in loader_stats.items():
                if hasattr(val, 'history') and getattr(val, 'has_new_data', True):
                    self.writer[loader_name].add_scalar(var_name, val.history[ind], epoch)

def wandb_write(stats: OrderedDict, epoch: int, ind=-1, prefix=''):
    for loader_name, loader_stats in stats.items():
        if loader_stats is None:
            continue
        wandblog = {}
        wandblog["epoch"] = epoch
        for var_name, val in loader_stats.items():
            if hasattr(val, 'history') and getattr(val, 'has_new_data', True):
                wandblog[prefix + var_name] = val.history[ind]
        wandb.log(wandblog)
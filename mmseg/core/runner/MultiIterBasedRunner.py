import os.path as osp
import platform
import shutil
import time
import warnings

import torch
from torch.optim import Optimizer

import mmcv
from mmcv.runner import IterBasedRunner, RUNNERS, get_host_info


class IterLoader:

    def __init__(self, dataloader, epoch=0):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = epoch

        if self._epoch != 0:
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


@RUNNERS.register_module()
class MultiIterBasedRunner(IterBasedRunner):
    def __init__(self, pre_iters, pre_loader_seed, **kwargs):
        super().__init__(**kwargs)

        self.pre_iters = pre_iters
        self.pre_loader_seed = pre_loader_seed


    @torch.no_grad()
    def pre_run_logic(self, data_loader, **kwargs):
        self.model.eval()
        self.logger.info(f'pre step STA: iter {self.pre_iters}; seed {self.pre_loader_seed}')

        if int(self.model.module.decode_head.iteration_counter) >= self.pre_iters:
            self.logger.info(f'resume checkpoints. skip pre step.')
            return

        self._pre_iter = 0
        while self._pre_iter < self.pre_iters:
            data_batch = next(data_loader)

            # * mem filled within decoder logic
            _ = self.model.val_step(data_batch, **kwargs)
            
            self._pre_iter += 1

        self.logger.info(f'pre step END: iter {self.pre_iters}; seed {self.pre_loader_seed}')

    def run(self, data_loaders, workflow, max_iters=None, **kwargs):
        """Start running.
        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)

        # * pre run to fill memory. skip pre step if there is only val step
        # * Note: this process will not be stored into state dict
        if not (len(workflow) == 1 and workflow[0][0] == 'val'):
            _pre_loader = IterLoader(data_loaders[0], epoch=self.pre_loader_seed) # * typically, is training
            self.pre_run_logic(_pre_loader, **kwargs)
            
            del _pre_loader

        self.call_hook('before_run')
        
        # * main run start
        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

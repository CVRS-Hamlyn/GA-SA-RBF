from __future__ import absolute_import, division, print_function

from trainer_MST import Trainer
from options import AutoFocusOptions
import torch.distributed as dist
import torch

options = AutoFocusOptions()
opts = options.parse()


if __name__== "__main__":
    trainer = Trainer(opts)
    trainer.train()

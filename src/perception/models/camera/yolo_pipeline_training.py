import os.path
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch

from src.perception.models.camera import train_backward
from src.perception.models.camera import train_forward
from src.perception.models.camera import detect

import yaml

from models.yolo import Model
from utils.general import check_git_info, check_amp
from utils.torch_utils import de_parallel, ModelEMA, smart_optimizer
from utils.metrics import fitness
from utils.loss import ComputeLoss


GIT_INFO = check_git_info()


class YoloPipeline:
    def __init__(self, cfg, save_path, device='cpu', hyp='/home/ubuntu/koopacar-system/src/perception/models/camera/data/hyps/hyp.scratch-low.yaml', optimizer='SGD'):
        self.device = device

        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                self.hyp = yaml.safe_load(f)

        self.model = Model(cfg, ch=3).to(self.device)
        self.model.hyp = self.hyp

        self.compute_loss = ComputeLoss(self.model)
        self.optimizer = smart_optimizer(self.model, optimizer, self.hyp['lr0'], self.hyp['momentum'], self.hyp['weight_decay'])

        self.best_fitness = 0
        self.ema = ModelEMA(self.model)
        self.safe_weights_path = os.path.join(save_path, 'weights')
        self.amp = check_amp(self.model)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        self._save_initial_model()

    def _save_initial_model(self):
        ckpt = {
            'epoch': 0,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'git': GIT_INFO,  # {remote, branch, commit} if a git repo
            'date': datetime.now().isoformat()}

        torch.save(ckpt, os.path.join(self.safe_weights_path, 'last.pt'))

    def forward(self, imgs, targets):
        # imgs = imgs.to(self.device, non_blocking=True).float()
        prediction = self.model(imgs)

        loss = self.compute_loss(prediction, targets.to(self.device))

        return loss

    def backward(self, loss, epoch, prediction):
        self.scaler.scale(loss).backward()

        # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)  # optimizer.step
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

        # save checkpoint
        fi = fitness(np.array(prediction).reshape(1, -1))
        if fi > self.best_fitness:
            self.best_fitness = fi

        ckpt = {
            'epoch': epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'git': GIT_INFO,  # {remote, branch, commit} if a git repo
            'date': datetime.now().isoformat()}

        torch.save(ckpt, os.path.join(self.safe_weights_path, 'last.pt'))


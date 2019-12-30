# --------------------------------------------------------------------------------------------------------
# 2019/12/26
# src - machine.py
# md
# --------------------------------------------------------------------------------------------------------
import pandas as pd
import torch.nn as nn
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

from configuration import rcp, cfg
from models.standard_models import MNSIT_Simple
from my_tools.python_tools import print_file, now_str
from my_tools.pytorch_tools import create_tb_summary_writer


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.cnn = MNSIT_Simple()
        # self.resnet = resnet18(pretrained=True)

    def forward(self, x):
        x = self.cnn(x)
        # x=self.resnet(x)
        return x


def run_training(model, train, valid, optimizer, loss):
    # DATA
    transform = transforms.Compose([transforms.ToPILImage(),
                                    # transforms.Resize(10),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
    train.transform, valid.transform = transform, transform
    train.save_csv(f'{cfg.log_path}train_df_{rcp.stage}.csv')
    valid.save_csv(f'{cfg.log_path}valid_df_{rcp.stage}.csv')
    train_loader = DataLoader(train, batch_size=rcp.bs, num_workers=8, shuffle=rcp.shuffle_batch)
    valid_loader = DataLoader(valid, batch_size=rcp.bs, num_workers=8, shuffle=rcp.shuffle_batch)

    print(f'# batches: train: {len(train_loader)}, valid: {len(valid_loader)}')

    trainer = create_supervised_trainer(model, optimizer, loss, device='cuda')
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'nll': Loss(loss)}, device='cuda')

    tb_writer = create_tb_summary_writer(model, train_loader, f'{rcp.tb_logdir}/{now_str()}')

    @trainer.on(Events.ITERATION_COMPLETED(every=int(1 + len(train_loader) / 120)))
    def print_dash(engine):
        print('-', sep='', end='', flush=True)

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_tenserboard(engine):
        tb_writer.add_scalar("batch/train/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        t_metrics = evaluator.state.metrics
        t_avg_acc = t_metrics['accuracy']
        t_avg_nll = t_metrics['nll']
        evaluator.run(valid_loader)
        v_metrics = evaluator.state.metrics
        v_avg_acc = v_metrics['accuracy']
        v_avg_nll = v_metrics['nll']
        print()
        print_file(f'{now_str("mm-dd hh:mm:ss")} |'
                   f'Ep:{engine.state.epoch:3} | '
                   f'Avg accuracy: {t_avg_acc:.5f}/{v_avg_acc:.5f} |  '
                   f'Avg loss: {t_avg_nll:.5f}/{v_avg_nll:.5f} | ',
                   f'{cfg.log_path}train_log_{rcp.stage}.txt')
        tb_writer.add_scalar("train/loss", t_avg_nll, engine.state.epoch)
        tb_writer.add_scalar("valid/loss", v_avg_nll, engine.state.epoch)
        tb_writer.add_scalar("train/acc", t_avg_acc, engine.state.epoch)
        tb_writer.add_scalar("valid/acc", v_avg_acc, engine.state.epoch)

    rcp.save_yaml()

    trainer.run(data=train_loader, max_epochs=rcp.max_epochs)
    return trainer

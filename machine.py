# --------------------------------------------------------------------------------------------------------
# 2019/12/26
# src - machine.py
# md
# --------------------------------------------------------------------------------------------------------
import pandas as pd
import torch.nn as nn
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, TensorboardLogger, WeightsHistHandler, OptimizerParamsHandler, WeightsScalarHandler, GradsScalarHandler, \
    GradsHistHandler
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events, engine, Engine, _prepare_batch
from ignite.metrics import Accuracy, Loss, Precision, Recall, TopKCategoricalAccuracy
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
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
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
                                                            'nll': Loss(loss),
                                                            'precision': Precision(average=True),
                                                            'recall': Recall(average=True),
                                                            'topK': TopKCategoricalAccuracy()}, device='cuda')

    # lr_scheduler = ExponentialLR(optimizer, gamma=0.975)
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    tb_logger = TensorboardLogger(log_dir=f'{rcp.tb_logdir}/{now_str()}')
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer, "lr"), event_name=Events.EPOCH_STARTED)
    tb_logger.attach(trainer, log_handler=WeightsHistHandler(model), event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(trainer, log_handler=WeightsScalarHandler(model), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=GradsScalarHandler(model), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=GradsHistHandler(model), event_name=Events.EPOCH_COMPLETED)
    tb_logger.close()

    @trainer.on(Events.ITERATION_COMPLETED(every=int(1 + len(train_loader) / 100)))
    def print_dash(engine):
        print('-', sep='', end='', flush=True)

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_tenserboard(engine):
        tb_logger.writer.add_scalar("batch/train/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        t_metrics = evaluator.state.metrics
        t_avg_acc = t_metrics['accuracy']
        t_avg_nll = t_metrics['nll']
        t_avg_prec = t_metrics['precision']
        t_avg_rec = t_metrics['recall']
        t_topk = t_metrics['topK']

        evaluator.run(valid_loader)
        v_metrics = evaluator.state.metrics
        v_avg_acc = v_metrics['accuracy']
        v_avg_nll = v_metrics['nll']
        v_avg_prec = v_metrics['precision']
        v_avg_rec = v_metrics['recall']
        v_topk = v_metrics['topK']
        lr_scheduler.step(v_avg_nll)
        print()
        print_file(f'{now_str("mm-dd hh:mm:ss")} |'
                   f'Ep:{engine.state.epoch:3} | '
                   f'acc: {t_avg_acc:.5f}/{v_avg_acc:.5f} | '
                   f'loss: {t_avg_nll:.5f}/{v_avg_nll:.5f} | '
                   f'prec: {t_avg_prec:.5f}/{v_avg_prec:.5f} | '
                   f'rec: {t_avg_rec:.5f}/{v_avg_rec:.5f} |'
                   f'topK: {t_topk:.5f}/{v_topk:.5f} |',
                   f'{cfg.log_path}train_log_{rcp.stage}.txt')

        tb_logger.writer.add_scalar("0_train/acc", t_avg_acc, engine.state.epoch)
        tb_logger.writer.add_scalar("0_train/loss", t_avg_nll, engine.state.epoch)
        tb_logger.writer.add_scalar("0_train/prec", t_avg_prec, engine.state.epoch)
        tb_logger.writer.add_scalar("0_train/rec", t_avg_rec, engine.state.epoch)
        tb_logger.writer.add_scalar("0_train/topK", t_topk, engine.state.epoch)

        tb_logger.writer.add_scalar("0_valid/acc", v_avg_acc, engine.state.epoch)
        tb_logger.writer.add_scalar("0_valid/loss", v_avg_nll, engine.state.epoch)
        tb_logger.writer.add_scalar("0_valid/prec", v_avg_prec, engine.state.epoch)
        tb_logger.writer.add_scalar("0_valid/rec", v_avg_rec, engine.state.epoch)
        tb_logger.writer.add_scalar("0_valid/topK", v_topk, engine.state.epoch)
        tb_logger.writer.flush()

    # trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: log_training_results(engine)) # This is the same as using a decorator

    rcp.save_yaml()
    trainer.run(data=train_loader, max_epochs=rcp.max_epochs)
    tb_logger.writer.close()
    return trainer


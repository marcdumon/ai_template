# --------------------------------------------------------------------------------------------------------
# 2019/12/26
# src - machine.py
# md
# --------------------------------------------------------------------------------------------------------
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18
import torch.nn as nn
from torchvision.transforms import transforms
from tqdm import tqdm
import torch as th
from models.standard_models import MNSIT_Simple




class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = MNSIT_Simple()

    def forward(self, x):
        x = self.cnn(x)

        if  state['epoch'] % 100==0:
            for name, param in self.cnn.conv2.named_parameters():
                # print(name,param)
                pass
            # print(state['epoch'])
        return x

state={'epoch':0}


def xxx(model, train_loader, val_loader, optimizer, loss):
    tb_logdir = '/media/md/Development/My_Projects/0_ml_project_template.v1/tensorboard/'
    trainer = create_supervised_trainer(model, optimizer, loss, device='cuda')
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'nll': Loss(loss)}, device='cuda')

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(initial=0, leave=False, total=len(train_loader),
                desc=desc.format(0))

    def create_summary_writer(model, data_loader, log_dir):
        writer = SummaryWriter(log_dir)
        data_loader_iter = iter(data_loader)
        x, y = next(data_loader_iter)
        x, y = x.to('cuda'), y.to('cuda')
        try:
            writer.add_graph(model, x)
        except Exception as e:
            print("Failed to save model graph: {}".format(e))
        return writer

    writer = create_summary_writer(model, train_loader, tb_logdir)

    @trainer.on(Events.ITERATION_COMPLETED(every=1))
    def log_training_loss(engine):
        pbar.desc = desc.format(engine.state.output)
        pbar.update(1)
        state['epoch']+=1
        if state['epoch'] %100==0:
            print('='*120)
            print(model.state_dict())
            th.save(model, 'xxx.pt')

    # def log_training_loss(trainer):
    # print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write("Train Results - Epoch: {}  Avg accuracy: {:.5f} Avg loss: {:.5f}"
                   .format(engine.state.epoch, avg_accuracy, avg_nll))

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_tenserboard(engine):
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Valid Results - Epoch: {}  Avg accuracy: {:.5f} Avg loss: {:.5f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

    return trainer

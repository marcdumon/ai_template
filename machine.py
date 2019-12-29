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

from configuration import cfg, rcp, original_rcp, original_cfg
from models.standard_models import MNSIT_Simple


class Model(nn.Module):
    '''
    Use this model to interfere in forward pass steps????
    '''

    def __init__(self):
        super(Model, self).__init__()
        self.cnn = MNSIT_Simple()
        # self.resnet = resnet18(pretrained=True)

    def forward(self, x):
        x = self.cnn(x)
        # x=self.resnet(x)
        # x2 = self.cnn.conv1(x)
        # x3 = self.resnet.layer1(x)

        # if state['epoch'] % 100 == 0:
        # for name, param in self.cnn.conv2.named_parameters():
        # print(name,param)
        # pass
        # print(state['epoch'])
        return x


model = Model()
# model.setup()

state = {'epoch': 0}


def run_training(model, train, valid, optimizer, loss):
    # DATA
    transform = transforms.Compose([transforms.ToPILImage(),
                                    # transforms.Resize(10),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    train.transform, valid.transform = transform, transform
    train_loader = DataLoader(train, batch_size=rcp.bs, num_workers=8, shuffle=rcp.shuffle_batch)
    valid_loader = DataLoader(valid, batch_size=rcp.bs, num_workers=8, shuffle=rcp.shuffle_batch)
    print('batches???', len(train_loader), len(valid_loader))

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

    writer = create_summary_writer(model, train_loader, rcp.tb_logdir)

    @trainer.on(Events.ITERATION_COMPLETED(every=1))
    def log_training_loss(engine):
        pbar.desc = desc.format(engine.state.output)
        pbar.update(1)
        # state['epoch'] += 1
        # if state['epoch'] % 100 == 0:
        #     print('=' * 120)
        #     print(model.state_dict())
        #     th.save(model, 'xxx.pt')

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
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        print("Valid Results - Epoch: {}  Avg accuracy: {:.5f} Avg loss: {:.5f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

    if rcp != original_rcp: rcp.save_yaml()
    if cfg != original_cfg: cfg.save_default_yaml()

    trainer.run(data=train_loader, max_epochs=rcp.max_epochs)
    return trainer


'''
ITERATION - loss: 0.10: 100%|██████████| 750/750 [00:12<00:00, 142.35it/s]Train Results - Epoch: 1  Avg accuracy: 0.98140 Avg loss: 0.05939
Valid Results - Epoch: 1  Avg accuracy: 0.97833 Avg loss: 0.07025
ITERATION - loss: 0.11: 1500it [00:25, 135.02it/s]Train Results - Epoch: 2  Avg accuracy: 0.98633 Avg loss: 0.04214
Valid Results - Epoch: 2  Avg accuracy: 0.98250 Avg loss: 0.05826
ITERATION - loss: 0.11: 2250it [00:38, 136.34it/s]Train Results - Epoch: 3  Avg accuracy: 0.98915 Avg loss: 0.03429
Valid Results - Epoch: 3  Avg accuracy: 0.98300 Avg loss: 0.06104
ITERATION - loss: 0.08: 3000it [00:51, 135.64it/s]Train Results - Epoch: 4  Avg accuracy: 0.98754 Avg loss: 0.03994
Valid Results - Epoch: 4  Avg accuracy: 0.97958 Avg loss: 0.06719
ITERATION - loss: 0.12: 3750it [01:03, 141.88it/s]Train Results - Epoch: 5  Avg accuracy: 0.99333 Avg loss: 0.02030
Valid Results - Epoch: 5  Avg accuracy: 0.98725 Avg loss: 0.04784
ITERATION - loss: 0.11: 4500it [01:16, 137.79it/s]Train Results - Epoch: 6  Avg accuracy: 0.99300 Avg loss: 0.02143
Valid Results - Epoch: 6  Avg accuracy: 0.98442 Avg loss: 0.05858
ITERATION - loss: 0.01: 5250it [01:29, 150.33it/s]Train Results - Epoch: 7  Avg accuracy: 0.99513 Avg loss: 0.01495
Valid Results - Epoch: 7  Avg accuracy: 0.98633 Avg loss: 0.05966
ITERATION - loss: 0.01: 6000it [01:42, 139.87it/s]Train Results - Epoch: 8  Avg accuracy: 0.99377 Avg loss: 0.02086
Valid Results - Epoch: 8  Avg accuracy: 0.98392 Avg loss: 0.07071
ITERATION - loss: 0.19: 6750it [01:55, 138.50it/s]Train Results - Epoch: 9  Avg accuracy: 0.99508 Avg loss: 0.01491
Valid Results - Epoch: 9  Avg accuracy: 0.98467 Avg loss: 0.07098
ITERATION - loss: 0.28: 7500it [02:08, 139.68it/s]Train Results - Epoch: 10  Avg accuracy: 0.99671 Avg loss: 0.01058
Valid Results - Epoch: 10  Avg accuracy: 0.98633 Avg loss: 0.06893
'''
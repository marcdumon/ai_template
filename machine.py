# --------------------------------------------------------------------------------------------------------
# 2019/12/26
# src - machine.py
# md
# --------------------------------------------------------------------------------------------------------
from distutils.dir_util import remove_tree, copy_tree
from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torchvision as thv
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss, Precision, Recall, TopKCategoricalAccuracy, ConfusionMatrix
from skimage import io
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from configuration import rcp, cfg
from models.standard_models import MNSIT_Simple
from my_tools.python_tools import print_file, now_str
from my_tools.pytorch_tools import DeNormalize
from visualization.confusion_matrix import pretty_plot_confusion_matrix
from visualization.make_graphviz_graph import make_dot


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
    # Data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomRotation(90),
        # transforms.Resize(10),
        # transforms.RandomVerticalFlip(.5),
        # transforms.RandomHorizontalFlip(.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train.transform, valid.transform = transform, transform
    train.save_csv(f'{rcp.base_path}train_df_{rcp.stage}.csv')
    valid.save_csv(f'{rcp.base_path}valid_df_{rcp.stage}.csv')
    train_loader = DataLoader(train, batch_size=rcp.bs, num_workers=8, shuffle=rcp.shuffle_batch)
    valid_loader = DataLoader(valid, batch_size=rcp.bs, num_workers=8, shuffle=rcp.shuffle_batch)
    print(f'# batches: train: {len(train_loader)}, valid: {len(valid_loader)}')

    # Save the graph.gv
    dot = make_dot(model(next(iter(train_loader))[0].to(cfg.device)), params=dict(model.named_parameters()))
    dot.render(f'{rcp.models_path}graph', './', format='png', cleanup=True)

    # Engines
    trainer = create_supervised_trainer(model, optimizer, loss, device=cfg.device)
    t_evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
                                                              'nll': Loss(loss),
                                                              'precision': Precision(average=True),
                                                              'recall': Recall(average=True),
                                                              'topK': TopKCategoricalAccuracy()}, device=cfg.device)
    v_evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
                                                              'nll': Loss(loss),
                                                              'precision_avg': Precision(average=True),
                                                              'recall_avg': Recall(average=True),
                                                              'topK': TopKCategoricalAccuracy(),
                                                              'conf_mat': ConfusionMatrix(num_classes=len(valid.classes), average=None),
                                                              }, device=cfg.device)

    # Tensorboard
    tb_logger = TensorboardLogger(log_dir=f'{rcp.tb_log_path}{rcp.stage}')
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer, "lr"), event_name=Events.EPOCH_STARTED)
    tb_logger.attach(trainer, log_handler=WeightsHistHandler(model), event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(trainer, log_handler=WeightsScalarHandler(model), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=GradsScalarHandler(model), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=GradsHistHandler(model), event_name=Events.EPOCH_COMPLETED)

    def score_function(engine):
        score = -1 * round(engine.state.metrics['nll'], 5)
        # score = engine.state.metrics['accuracy']
        return score

    # Schelulers
    # lr_scheduler = ExponentialLR(optimizer, gamma=0.975)
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=.5, min_lr=1e-7, verbose=True)
    es_handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
    v_evaluator.add_event_handler(Events.COMPLETED, es_handler)

    # Checkpoint
    to_save = to_load = {'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    save_best = Checkpoint(to_save, DiskSaver(f'{rcp.models_path}', require_empty=False, create_dir=True),
                           n_saved=4, filename_prefix=f'best_{rcp.stage}',
                           score_function=score_function, score_name="val_loss",
                           global_step_transform=global_step_from_engine(trainer))

    v_evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), save_best)
    load_checkpoint = False

    if load_checkpoint:
        resume_epoch = 6
        cp = f'{rcp.models_path}last_{rcp.stage}_checkpoint.pth'
        obj = th.load(f'{cp}')
        Checkpoint.load_objects(to_load, obj)

        @trainer.on(Events.STARTED)
        def resume_training(engine):
            engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
            engine.state.epoch = resume_epoch - 1

    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def save_last_checkpoint(engine):
        checkpoint = {}
        to_save = to_load = {'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        for k, obj in to_save.items():
            checkpoint[k] = obj.state_dict()
        th.save(checkpoint, f'{rcp.models_path}last_{rcp.stage}_checkpoint.pth')

    @trainer.on(Events.STARTED)
    def show_batch_images(engine):
        images, labels = next(iter(train_loader))
        denormalize = DeNormalize((0.1307,), (0.3081,))
        for i in range(len(images)):
            images[i] = denormalize(images[i])
        images = images.to(cfg.device)
        grid = thv.utils.make_grid(images)
        tb_logger.writer.add_image('images', grid, 0)
        tb_logger.writer.add_graph(model, images)
        tb_logger.writer.flush()

    # @trainer.on(Events.ITERATION_COMPLETED(every=10))
    # def log_tenserboard(engine):
    #     tb_logger.writer.add_scalar("batch/train/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.ITERATION_COMPLETED(every=int(1 + len(train_loader) / 100)))
    def print_dash(engine):
        print('-', sep='', end='', flush=True)

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def get_top_losses(engine, k=6):
        nll_loss = nn.NLLLoss(reduction='none')
        df = predict_dataset(model, valid, nll_loss, transform, bs=rcp.bs * 10, device=cfg.device)
        df.sort_values('loss', ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        for i, row in df.iterrows():
            img = io.imread(row['fname'], as_gray=False)
            img = th.as_tensor(img[np.newaxis, :, :])  # add C
            tag = f'TopLoss_{engine.state.epoch}/{row.loss:.4f}/{row.target}/{row.pred}/{row.pred2}'
            tb_logger.writer.add_image(tag, img, 0)
            if i >= k - 1: break
        tb_logger.writer.flush()

    valid_dl_sorted = DataLoader(valid, batch_size=rcp.bs, shuffle=False)  # Can't use valid_dl because impossible te know the indices for a batch when shuffle=True

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_valid_results(engine):
        t_evaluator.run(train_loader)
        t_metrics = t_evaluator.state.metrics
        t_avg_acc = t_metrics['accuracy']
        t_avg_nll = t_metrics['nll']
        t_avg_prec = t_metrics['precision']
        t_avg_rec = t_metrics['recall']
        t_topk = t_metrics['topK']

        v_evaluator.run(valid_loader)
        v_metrics = v_evaluator.state.metrics
        v_avg_acc = v_metrics['accuracy']
        v_avg_nll = v_metrics['nll']
        v_avg_prec = v_metrics['precision_avg']
        v_avg_rec = v_metrics['recall_avg']
        v_topk = v_metrics['topK']
        # lr_scheduler.step(v_avg_nll)  # ReduceLROnPlateau
        print()
        print_file(f'{now_str("mm-dd hh:mm:ss")} |'
                   f'Ep:{engine.state.epoch:3} | '
                   f'acc: {t_avg_acc:.5f}/{v_avg_acc:.5f} | '
                   f'loss: {t_avg_nll:.5f}/{v_avg_nll:.5f} | '
                   f'prec: {t_avg_prec:.5f}/{v_avg_prec:.5f} | '
                   f'rec: {t_avg_rec:.5f}/{v_avg_rec:.5f} |'
                   f'topK: {t_topk:.5f}/{v_topk:.5f} |',
                   f'{rcp.results_path}train_log_{rcp.stage}.txt')
        tb_logger.writer.add_scalar("0_train/acc", t_avg_acc, engine.state.epoch)
        tb_logger.writer.add_scalar("0_train/loss", t_avg_nll, engine.state.epoch)
        tb_logger.writer.add_scalar("0_train/prec", t_avg_prec, engine.state.epoch)
        tb_logger.writer.add_scalar("0_train/rec", t_avg_rec, engine.state.epoch)
        tb_logger.writer.add_scalar("0_train/topK", t_topk, engine.state.epoch)
        tb_logger.writer.flush()
        tb_logger.writer.add_scalar("0_valid/acc", v_avg_acc, engine.state.epoch)
        tb_logger.writer.add_scalar("0_valid/loss", v_avg_nll, engine.state.epoch)
        tb_logger.writer.add_scalar("0_valid/prec", v_avg_prec, engine.state.epoch)
        tb_logger.writer.add_scalar("0_valid/rec", v_avg_rec, engine.state.epoch)
        tb_logger.writer.add_scalar("0_valid/topK", v_topk, engine.state.epoch)
        tb_logger.writer.flush()

        # Confusion Matrix
        if trainer.state.epoch % 5 == 0:
            cm = v_metrics['conf_mat']
            cm_df = pd.DataFrame(cm.numpy(), index=valid.classes, columns=valid.classes)
            pretty_plot_confusion_matrix(cm_df, f'{rcp.results_path}cm_{rcp.stage}_{trainer.state.epoch}.png', False)

    # Tensorboard Projector
    # helper function
    def select_n_random(ds, n=100):
        """Selects n random datapoints and their corresponding labels from a dataset"""
        perm = th.randperm(len(ds))
        perm = perm[:n]
        imgs = th.stack([ds[i][0] for i in perm])
        lbls = [ds[i][1] for i in perm]
        return imgs, lbls

    # select random images and their target indices
    images, labels = select_n_random(train, 100)
    # get the class labels for each image
    class_labels = [train.classes[lab] for lab in labels]
    # log embeddings
    features = images.view(-1, images.shape[-1] * images.shape[-2])  # nx1x28x28 -> n*784
    tb_logger.writer.add_embedding(features, metadata=class_labels, label_img=images)

    @trainer.on(Events.EPOCH_COMPLETED(every=5))
    def log_pr_curve(engine):
        # 1. gets the probability predictions in a test_size x num_classes Tensor
        # 2. gets the preds in a test_size Tensor
        # takes ~10 seconds to run
        class_probs = []
        class_preds = []
        with th.no_grad():
            for data in valid_loader:
                images, labels = data
                images, labels = images.to(cfg.device), labels.to(cfg.device)
                output = model(images)
                class_probs_batch = [th.softmax(el, dim=0) for el in output]
                _, class_preds_batch = th.max(output, 1)

                class_probs.append(class_probs_batch)
                class_preds.append(class_preds_batch)
        test_probs = th.cat([th.stack(batch) for batch in class_probs])
        test_preds = th.cat(class_preds)

        # helper function
        def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
            '''
            Takes in a "class_index" from 0 to 9 and plots the corresponding
            precision-recall curve
            '''
            tensorboard_preds = test_preds == class_index
            tensorboard_probs = test_probs[:, class_index]

            tb_logger.writer.add_pr_curve(valid.classes[class_index],
                                          tensorboard_preds,
                                          tensorboard_probs,
                                          global_step=global_step)
            tb_logger.writer.close()

        # plot all the pr curves
        for i in range(len(valid.classes)):
            add_pr_curve_tensorboard(i, test_probs, test_preds, engine.state.epoch)

    cfg.save_yaml()
    rcp.save_yaml()
    trainer.run(data=train_loader, max_epochs=rcp.max_epochs)
    tb_logger.writer.close()

    return trainer


def predict_dataset(model, dataset, loss_fn, transform=None, bs=32, device=cfg.device):
    """
    Takes a model, dataset and loss_fn returns a dataframe with columns = [fname, targets, loss, pred]
    """
    if transform:
        dataset.transform = transform
    else:
        dataset.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize(10),
            # transforms.RandomVerticalFlip(.5),
            # transforms.RandomHorizontalFlip(.5),
            transforms.ToTensor(),  # (H x W x C) in the range [0, 255] -> (C x H x W) in the range [0.0, 1.0]
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataloader = DataLoader(dataset, bs, shuffle=False, num_workers=8)
    df = pd.DataFrame()
    df['fname'] = dataset.data
    df['target'] = dataset.targets
    model.to(device)
    loss = []
    pred = []
    pred2 = []
    for image, target in dataloader:
        model.eval()
        with th.no_grad():
            image, target = image.to(device), target.to(device)
            logits = model(image)
            l = loss_fn(logits, target)
            p = th.argmax(logits, dim=1)
            # 2nd argmax
            p2 = th.topk(logits, 2, dim=1)  # returns namedtuple (values, indices)
            p2 = p2.indices[:, 1]  # second column
            loss += list(l.to('cpu').numpy())
            pred += list(p.to('cpu').numpy())
            pred2 += list(p2.to('cpu').numpy())
    df['loss'] = loss
    df['pred'] = pred
    df['pred2'] = pred2
    return df


def setup_experiment():
    """
    Create directories for experiment:
        ../temp_reports
            /experiment
                /yyyymmdd_hhmmss
                    /models
                    /results
                    /src

    """
    # Create paths
    Path(f'{rcp.models_path}').mkdir(parents=True, exist_ok=True)
    Path(f'{rcp.results_path}').mkdir(parents=True, exist_ok=True)
    # Path(f'{rcp.src_path}').mkdir(parents=True, exist_ok=True)
    # Copy src
    source = '../src'
    destination = f'{rcp.src_path}'
    # remove_tree(destination)  # copy_tree can't overwrite
    copy_tree(source, destination)


def close_experiment(experiment: str, datetime: str):
    """
    move experiment to ../reports
    move ../tensorboard/experiment to ../reports
    """
    source = f'{cfg.temp_report_path}{experiment}/{datetime}/'
    tb_source = f'../tensorboard/{experiment}/{datetime}/'
    destination = f'../reports/{experiment}/{datetime}/'
    copy_tree(source, destination, verbose=2)
    copy_tree(tb_source, f'{destination}tensorboard', verbose=2)
    remove_tree(source, verbose=2)
    remove_tree(tb_source, verbose=2)
    exit()


if __name__ == '__main__':
    # close_experiment('baseline', '20200103_230746')
    pass

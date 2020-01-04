## Todo
Short Term:
- make baseline for MNIST: train simple cnn and get results for testset. => freeze baseline and don't use test anymore
- run first experiment: what's the effect of increasing noisy labels.
- check get_mean_and_std(dataset) from pytorch_tools

 Mid Term (goals version 0.2):
- sanity checking before training to catch crashes early
- lr-scheduler: check LambdaLR()
- check simulate_values and plot_values in param_scheduler.py
- save description and print(model)
- organise lr_finder
- merge Config and Recipe ? 
- check docstring inheritance in children (fi in standard_dataset)
- manage model parameters (like fc-size,...)
- create_tb_summary_writer() creates tb_summary writer and graph. split this
- move load_checkpoint outside run_training
- show_batch_images and tb projector not for every stage (same)
- confusion matrix outside? also for last epoch
- show_mpl_grid in pytorch_tools: rename to mpl_show_grid and make generic for tensors, list, df,...
 


- Check:
    class MnistResNet(ResNet):
    def __init__(self):
   ---> super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self, x):
    ===> return torch.softmax(super(MnistResNet, self).forward(x), dim=-1)
    from: https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial/ 
    But bigg mistake in loss (see: https://github.com/marrrcin/pytorch-resnet-mnist/issues/1)
- check: https://pytorch.org/blog/towards-reproducible-research-with-pytorch-hub/



- use \_\_init\_\_.py to have better imports elsewhere? See: [why would I put python code in __init__.py files
- check python logging
- refactor with Abstract Classes (framework design pattern or mixin classes)? 
See: [Raymond Hettinger «Build powerful, new data structures with Python's abstract base classes](https://www.youtube.com/watch?v=S_ipdVNSFlo).
- what is TopKCategoricalAccuracy?
- layer visualisation: https://github.com/utkuozbulak/pytorch-cnn-visualizations
- https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict/blob/master/notebook.ipynb
- doctests?
- put all imports in \_\_init\_\_().  According to Raymond Hettinger - Beyond PEP 8, we don't need from aaa.bbb.ccc import ddd
 [see](https://stackoverflow.com/questions/5831148/why-would-i-put-python-code-in-init-py-files/5831225)

- package to attach(not install, but use to implement myself):
    - [pretrained_models](https://github.com/Cadene/pretrained-models.pytorch)
    - [LR_Finder](https://github.com/davidtvs/pytorch-lr-finder/blob/master/torch_lr_finder/lr_finder.py)
    - [High-Level Training, Data Augmentation, and Utilities for Pytorch](https://github.com/ncullen93/torchsample)
- GPU memory [see fast ai](https://github.com/fastai/fastai/blob/master/fastai/utils/mem.py)

Long Term:
- Check out: https://github.com/FrancescoSaverioZuppichini/mirror
- Install automatic API documentation: [epydoc](http://epydoc.sourceforge.net/)
- Check ODO: https://github.com/blaze/odo
- Install Sphinx for documentation building
- Checkout this animation about convolution : [Convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic)
## Todo
Short Term:
- lr-finder
- sanity checking before training to catch crashes early
- finish logging
- Make baseline for MNIST: train simple cnn and get results for testset. => freeze baseline and don't use test anymore
- run first experiment: what's the effect of increasing noisy labels.
- save_tb: copy the tb results to ../reports




 
 
 Mid Term:
- use \_\_init\_\_.py to have better imports elsewhere? See: [why would I put python code in __init__.py files
- check python logging
- refactor with Abstract Classes (framework design pattern or mixin classes)? 
See: [Raymond Hettinger «Build powerful, new data structures with Python's abstract base classes](https://www.youtube.com/watch?v=S_ipdVNSFlo).
- what is TopKCategoricalAccuracy?
- layer visualisation: https://github.com/utkuozbulak/pytorch-cnn-visualizations
- https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict/blob/master/notebook.ipynb

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
## Todo
Short Term:
- create a dataset base class that every dataset should be inherited with paths, 
classes, etc. When I add methods in the furure, all dataset classes will automatically have them. When I use certain parameters like classes etc somewhere else, it workks fror all 
datasets.
- package to attach(not install, but use to implement myself):
    - [pretrained_models](https://github.com/Cadene/pretrained-models.pytorch)
    - [LR_Finder](https://github.com/davidtvs/pytorch-lr-finder/blob/master/torch_lr_finder/lr_finder.py)
    - [High-Level Training, Data Augmentation, and Utilities for Pytorch](https://github.com/ncullen93/torchsample)
- visualisation of batches and results with torchvision.utils.make_grid()
- use \_\_init\_\_.py to have better imports elsewhere? See: [why would I put python code in __init__.py files
](https://stackoverflow.com/questions/5831148/why-would-i-put-python-code-in-init-py-files/5831225)
 - sanity checking before training to catch crashes early
 - put all imports in \_\_init\_\_().  According to Raymond Hettinger - Beyond PEP 8, we don't need from aaa.bbb.ccc import ddd
 
 
 
 
 
 Mid Term:
- refactor with Abstract Classes (framework design pattern or mixin classes)? See: [Raymond Hettinger «Build powerful, new data structures with Python's abstract base classes](https://www.youtube.com/watch?v=S_ipdVNSFlo).
- refactor with Data classes? See [- refactor with Data classes? See (Raymond Hettinger - Dataclasses: The code generator to end all code generators)
](https://www.youtube.com/watch?v=T-TwcmT6Rcw)



Long Term:
- Install automatic API documentation: [epydoc](http://epydoc.sourceforge.net/)
- Check ODO: https://github.com/blaze/odo
- Install Sphinx for documentation building

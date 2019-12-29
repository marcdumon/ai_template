# --------------------------------------------------------------------------------------------------------
# 2019/12/28
# src - checkout.py.py
# md
# --------------------------------------------------------------------------------------------------------
'''
Checkout an experiment.
Features:
    - it schould be possible to redo the experiment, by simply go to it's source and run main_app.py
    - it schould be easy to compare the results from different experiments

Needs to store:
    - src
    - cfg, rcp
    - epoch log (train/valid loss, lr's, ...)
    - dataset dataframes to know the exact data used
    - DON't change parameter in code, they will not be saved
    - best, last x models
    - tensorboard logs
'''




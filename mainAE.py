import numpy as np
import os, json, cv2, random
import yaml
import itertools
import pandas as pd
import pathlib
from trainDetectron import run_training_pipeline

def main():
    ind_params = {
        'modalidad': 'FLAIR',
        'modelo': 'mask_rcnn_R_101_FPN_3x.yaml',
        'flip': 'vertical',
        'batch_size': 2,
        'gamma': 0.05,
        'base_lr': 0.001,
        'weight_decay': 0.0001,
        'maxiter_steps': 5000
    }

    run_training_pipeline(ind_params)

if __name__ == "__main__":
    main()
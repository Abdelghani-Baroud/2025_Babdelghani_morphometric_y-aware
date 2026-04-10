import os
import numpy as np
import pandas as pd
from soma import aims, aimsalgo


"""
Preprocess data for contrastive after deepfolding skeleton and
foldlabel generation.
1) Generate disbottom dataset: should disbottom
be generated before cropping ?
2) Mask resampled foldlabels
"""
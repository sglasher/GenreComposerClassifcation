# Composer Classification w/ MIDI Files

This project aims to classify composers based on MIDI files of their compositions using Convolutional Neural Networks (CNNs) and Long Short-Term Memory networks (LSTMs). By leveraging the unique patterns present in MIDI data, the model can learn to distinguish the distinctive styles of different composers.

## Objective

This project involves applying both CNNs and LSTMs separately to MIDI data for composer classification. By independently evaluating the performance of these two neural network architectures, we aim to identify the most effective approach for this specific task.

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Installation](#installation)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

## Background

Music classification and analysis have seen significant advancements with the adoption of machine learning techniques. Using MIDI files for classification provides a unique challenge due to the combination of sequential and structural patterns present in these files. This project investigates the utility of CNNs and LSTMs in capturing these patterns for composer identification.

## Dataset

The dataset consists of MIDI files contributed by various composers, including Beethoven, Mozart, Bach, and Chopin. The MIDI files are organized into three main folders: "dev," "test," and "train." Each of these folders contains subfolders dedicated to each composer, storing their respective MIDI files.

- `dev/composerX`: MIDI files for composer X
- `dev/composerY`: MIDI files for composer Y

- `test/composerX`: MIDI files for composer X
- `test/composerY`: MIDI files for composer Y

- `train/composerX`: MIDI files for composer X
- `train/composerY`: MIDI files for composer Y

The dataset is composed of MIDI files from multiple composers. These MIDI files were preprocessed into representations that are suitable for training the neural networks, such as piano rolls.

## Installation

1. Clone the repository: https://github.com/sglasher/GenreComposerClassifcation.git
2. Navigate to the project directory: cd GenreComposerClassification
3. Ensure you have Python 3.7 or later installed.

## Python Libraries Used

The project relies on the following Python libraries for various tasks:

```
from google.colab import drive
import os
import shutil
import mido
import numpy as np
from keras.utils import to_categorical
import pretty_midi
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
 LabelEncoder, OneHotEncoder, normalize, MinMaxScaler,
 StandardScaler
)
from sklearn.metrics import (
 accuracy_score, precision_score, recall_score, f1_score,
 roc_auc_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
 Conv2D, AveragePooling2D,
 MaxPooling2D, Flatten,
 Dense, LSTM, Dropout,
 RepeatVector, Activation
)
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from skimage.transform import resize

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import keras_tuner
```

## Results

Model evaluation represents the concluding phase in the architecture, assessing the model's effectiveness on unseen data. The primary objective is to gauge both model accuracy and its ability to generalize beyond the training set. An established classification metric, accuracy, quantifies correct predictions relative to total instances evaluated. The model achieved an overall accuracy of approximately 40%.

To comprehensively evaluate the model's performance, we employed precision, recall, and ROC AUC. These metrics offer insights, particularly when dealing with class imbalances or nuanced implications of false positives/negatives. Precision assesses true positives within total positive predictions. Recall calculates accurately identified actual positives. Our ultimate evaluation metric, ROC AUC, examines classifier performance across varying thresholds – a score of 1 signifies a perfect model, while 0.5 denotes random chance.

Distinct composers exhibit varying performance metrics. Bach, for instance, displays perfect precision alongside a 50% recall, accompanied by an ROC AUC of 0.75. Bartok's precision and recall stand at 0.5 and 0.25 respectively, with an ROC AUC of approximately 0.609. Byrd demonstrates notable precision (0.667), perfect recall, and an ROC AUC of around 0.9677. Chopin, Hummel, Mendelssohn, and Mozart showcase diverse precision and recall values, with ROC AUC ranging from 0.669 to 0.685. Conversely, Handel and Schumann register a precision and recall of 0.0, paired with ROC AUC values of roughly 0.4516 and 0.4844 respectively.

While the model's effectiveness exhibits variability, the overarching accuracy suggests significant room for improvement. The range of performance metrics underscores the importance of conducting a thorough model evaluation, employing multiple metrics to gain a comprehensive grasp of the model's strengths and weaknesses.

The distinctive outcomes for individual composers underscore the need for in-depth investigation into factors contributing to suboptimal performance. These factors may encompass training data quality, quantity, or inherent complexities tied to differentiating musical styles.

In summary, though the model's effectiveness fluctuates across composers, the overall accuracy indicates substantial scope for enhancement. A comprehensive evaluation employing diverse metrics unveils intricate model characteristics, guiding future optimization endeavors.

## Future Enhancements

- Investigate hybrid architectures combining CNNs and LSTMs for improved classification accuracy.
- Explore methods to mitigate overfitting and enhance model generalization.
- Consider incorporating data augmentation techniques for improved model robustness.

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

Model evaluation constitutes the final phase in the architecture, determining the model's efficacy on unseen data. The main aim is to ensure the model possesses adequate accuracy for deployment and the ability to generalize beyond the training dataset. A common metric for classification problems, as seen in our identified issue, is the model's accuracy - the ratio of correct predictions to the total evaluated instances. The analysis revealed an overall accuracy of around 40%.

To conduct a comprehensive evaluation of the underlying model, precision, recall, and ROC AUC were employed. This offers a holistic perspective, especially in scenarios with class imbalances or varying consequences of false positives/negatives based on the application. Precision gauges the ratio of true positive predictions to total positive predictions. Recall quantifies the proportion of actual positives correctly identified. ROC AUC, our final evaluation metric, provides insights into a binary classifier's performance as the discrimination threshold varies - a score of 1 signifies a perfect model, while 0.5 implies no better performance than random chance.

The presented results highlight distinct performance metrics for different composers. For instance, Bach achieves a perfect precision of 1.0, signifying accurate predictions for this composer. However, its recall stands at 0.5, indicating only half of Bach's actual instances were correctly classified. Its ROC AUC is 0.75. Bartok exhibits a precision of 0.5, a recall of 0.25, and an ROC AUC of about 0.609. This suggests that while half of Bartok's predictions were correct, only a quarter of actual instances were accurately detected. Byrd's metrics are particularly remarkable, with a precision of approximately 0.667 and a recall of 1.0, implying identification of all actual Byrd instances. Moreover, an ROC AUC of roughly 0.9677 underscores the model's competence with Byrd's data. Chopin, Hummel, Mendelssohn, and Mozart demonstrate varying levels of precision and recall, ranging from approximately 0.286 to 0.5 for precision, and 0.25 to 0.5 for recall. Their ROC AUC values range between 0.669 and 0.685. In contrast, Handel and Schumann both exhibit a precision and recall of 0.0. This indicates the model's inability to correctly classify any of their compositions. Their ROC AUC values (about 0.4516 for Handel and 0.4844 for Schumann) further emphasize the model's inadequate performance with their data. Such outcomes might necessitate deeper investigation into reasons for suboptimal performance, which could encompass aspects like training data quality and quantity for these composers or inherent intricacies in distinguishing their musical styles.

While the model displays varying degrees of effectiveness across composers, its overall accuracy underscores significant potential for improvement. The diverse performance metrics underscore the importance of a comprehensive model evaluation, considering multiple metrics to obtain a thorough understanding of the model's strengths and limitations.

## Future Enhancements

- Investigate hybrid architectures combining CNNs and LSTMs for improved classification accuracy.
- Explore methods to mitigate overfitting and enhance model generalization.
- Consider incorporating data augmentation techniques for improved model robustness.

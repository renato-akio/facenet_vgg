import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from random import choice, sample
import cv2
from imageio import imread
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras import regularizers
from keras.layers import Input, Embedding, LSTM, Dropout, BatchNormalization,Dense, concatenate, Flatten, Conv1D
from keras.optimizers import RMSprop, Adam
import test as test
import warnings
warnings.filterwarnings("ignore")
#%matplotlib inline

from keras_vggface.vggface import VGGFace
from glob import glob
from keras import backend as K
from keras.preprocessing import image
from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D, Lambda, Reshape
from collections import defaultdict
from keras_vggface.utils import preprocess_input

TRAIN_BASE = 'train'
all_images = glob(TRAIN_BASE + "*/*/*/*.jpg")

#folders with name F09 will be our validation dataset and the rest will be in train dataset
val_families = "F09"
train_images = [x for x in all_images if val_families not in x]
val_images = [x for x in all_images if val_families in x]

ppl = [x.split("\\")[-3] + "\\" + x.split("\\")[-2] for x in all_images]

#preparing train and test dataset
train_person_to_images_map = defaultdict(list)

for x in train_images:
    train_person_to_images_map[x.split("\\")[-3] + "\\" + x.split("\\")[-2]].append(x)

val_person_to_images_map = defaultdict(list)

for x in val_images:
    val_person_to_images_map[x.split("\\")[-3] + "\\" + x.split("\\")[-2]].append(x)

relationships = pd.read_csv('train_relationships_custom.csv')
relationships = list(zip(relationships.p1.values, relationships.p2.values))
relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

train = [x for x in relationships if val_families not in x[0]]
val = [x for x in relationships if val_families in x[0]]

#loading facenet model
model_path = 'facenet_keras.h5'
facenet_model = load_model(model_path)

#We will train full network except the last 3 layers
for layer in facenet_model.layers[:-3]:
    layer.trainable = True

#We will train full network except the last 3 layers
vgg_model = VGGFace(model='resnet50', include_top=False)
for layer in vgg_model.layers[:-3]:
    layer.trainable = True

valx = test.gen(val, val_person_to_images_map, batch_size=100)

for i in valx:
    valx = i
    break

model = test.model(facenet_model, vgg_model)

model.compile(loss="binary_crossentropy", metrics=[test.auc], optimizer=Adam(1e-5))

model.summary()

# Training the model and saving it with name facenet_vgg.h5***

import datetime
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Clear any logs from previous runs
# !rm -rf ./logs/

log_dir = "logs"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=10)

checkpoint = ModelCheckpoint('facenetvgg.h5', monitor='val_auc', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.1, patience=20, verbose=1)

callbacks_list = [tensorboard_callback, checkpoint, reduce_on_plateau, es]

if __name__ ==  '__main__':
    history = model.fit_generator(test.gen(train, train_person_to_images_map, batch_size=16), use_multiprocessing=True,
                              validation_data=(valx[0], valx[1]), epochs=50, verbose=1,
                              workers=0, callbacks=callbacks_list, steps_per_epoch=200)
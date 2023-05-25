#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout,MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from keras.applications.imagenet_utils import decode_predictions 
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


train_data = []
train_labels = [] 


# In[3]:


for directory_path in glob.glob("Dataset/Full Image/Training Data/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (289, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_data.append(img)
        train_labels.append(label)


# In[4]:


train_data = np.array(train_data)
train_labels = np.array(train_labels)


# In[5]:


val_data = []
val_labels = [] 


# In[6]:


for directory_path in glob.glob("Dataset/Full Image/Validation Data/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (289, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        val_data.append(img)
        val_labels.append(label)


# In[7]:


val_data = np.array(val_data)
val_labels = np.array(val_labels)


# In[8]:


le = preprocessing.LabelEncoder()


# In[9]:


le.fit(train_labels)
train_label = le.transform(train_labels)


# In[10]:


le.fit(val_labels)
val_label = le.transform(val_labels)


# In[11]:


train_data.shape


# In[12]:


train_data = train_data.reshape(train_data.shape[0],train_data.shape[2],train_data.shape[1],train_data.shape[3])


# In[13]:


val_data = val_data.reshape(val_data.shape[0],val_data.shape[2],val_data.shape[1],val_data.shape[3])


# In[14]:


train_data.shape


# In[15]:


vgg_model = Sequential()
pre_trained_model = tf.keras.applications.VGG16(input_shape=(289, 188,3), weights='imagenet', include_top=False,pooling='avg')


# In[16]:


for layer in vgg_model.layers:
    layer.trainable = False


# In[17]:


vgg_model.add(pre_trained_model)


# In[18]:


vgg_model.add(layers.Flatten())
vgg_model.add(layers.Dense(512, activation='relu'))
vgg_model.add(layers.Dense(1, activation='sigmoid'))


# In[19]:


vgg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[20]:


history = vgg_model.fit(train_data, train_label, epochs=20,  validation_data= (val_data, val_label))


# In[ ]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = vgg_model.evaluate(val_data, verbose=2)


# In[ ]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


# In[ ]:


predict = vgg_model.predict(val_data)


# In[ ]:


val_predict = ((predict > 0.5)+0).ravel()


# In[ ]:


print(classification_report(val_label, val_predict))


# In[ ]:


print(confusion_matrix(val_label, val_predict))


# In[ ]:


train_loss, train_acc = vgg_model.evaluate(train_data, verbose=2)
print(train_acc)

val_loss, val_acc = vgg_model.evaluate(val_data, verbose=2)
print(val_acc)


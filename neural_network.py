
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
from tensorflow import set_random_seed


# In[2]:


df=pd.read_csv("Final_data.csv")
print(df.shape)
df.head()


# In[3]:


ids=df['air_store_id'].tolist()


# In[4]:


date=df['visit_date'].tolist()
x=[i.split("-") for i in date]
date=[int(i[2]) for i in x]


# In[5]:


dataset = df.copy()
dataset.pop('visit_date')
dataset.pop('min_visitors')
dataset.pop('median_visitors')
dataset.pop('max_visitors')
dataset.pop('count_observations')

# adding date to dataset
s = pd.Series(date)
df1 = pd.DataFrame({'date':s})
dataset=dataset.join(df1)

#adding store id
s = pd.Series(ids)
labels, levels = pd.factorize(s)
df1 = pd.DataFrame({'air_store_id':(labels)})
dataset.pop('air_store_id')
dataset=dataset.join(df1)


# In[6]:


train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# In[7]:


train_labels = train_dataset.pop('visitors')
test_labels = test_dataset.pop('visitors')


# In[8]:


train_stats = train_dataset.describe()
train_stats=train_stats.transpose()


# In[9]:


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# In[10]:


def build_model():
    model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(normed_train_data.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(l=0.1)),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dropout(0.1),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1, activation=tf.nn.relu)
  ])

    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


# In[11]:


model = build_model()
model.load_weights('neural_network.h5')


# In[58]:


# Display training progress by printing a single dot for each completed epoch
tic=time.time()
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 200

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[PrintDot()])
toc=time.time()


# In[14]:


loss, mae, mse = model.evaluate(normed_test_data,test_labels, verbose=0)

print("Testing set Mean abs Error: {:5.2f}".format(mae))
print("Time taken for training:"+str(toc-tic))


# In[33]:


tf.keras.models.save_model(model,"./neural_network.h5",overwrite=True,include_optimizer=True)


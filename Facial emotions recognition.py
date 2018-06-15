
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


# ## Useful functions

# In[ ]:


def save_model(model, filename):
    model_json = model.to_json()
    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename + ".h5")
    print("Saved model to disk")


# ## Load data

# In[ ]:


DATA_FOLDER = "./data/"


# In[ ]:


df_training = pd.read_parquet(DATA_FOLDER + "training.parquet")


# In[ ]:


df_public_test = pd.read_parquet(DATA_FOLDER + "public_test.parquet")


# In[ ]:


df_private_test = pd.read_parquet(DATA_FOLDER + "private_test.parquet")


# ## Process data

# In[ ]:


x_train = df_training.as_matrix()[:,1:]
y_train = df_training.as_matrix()[:,0]
x_test = df_public_test.as_matrix()[:,1:]
y_test = df_public_test.as_matrix()[:,0]


# In[ ]:


x_train = x_train.reshape(-1, 48, 48, 1)
x_test = x_test.reshape(-1, 48, 48, 1)


# In[ ]:


x_train = x_train.astype("float32")/255.
x_test = x_test.astype("float32")/255.


# In[ ]:


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# ## Create model

# In[ ]:


model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',
                 input_shape = (48, 48, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


# In[ ]:


datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])


# In[ ]:


annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)


# In[ ]:


hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                           steps_per_epoch=500,
                           epochs=20, #Increase this when not on Kaggle kernel
                           validation_data=(x_test[:400,:], y_test[:400,:]), #For speed
                           callbacks=[annealer])


# In[ ]:


final_loss, final_acc = model.evaluate(x_test, y_test, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))


# In[ ]:


save_model(model, "./trained_models/first_model")


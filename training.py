import pickle
import numpy as np
import tensorflow as tf

x=pickle.load(open('one_shot_pre_data.pickle','rb'))
y=pickle.load(open('one_shot_pre_label.pickle','rb'))
z=pickle.load(open('one_shot_pre_data_2.pickle','rb'))

print(np.shape(y))
print(np.shape(x))
print(np.shape(z))
import keras
filename="metrics.csv"
es=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, mode='auto', baseline=None, restore_best_weights=True)
csv=keras.callbacks.CSVLogger(filename, separator=',', append=False)
checkpoint=keras.callbacks.ModelCheckpoint("checkpoints/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


x=np.swapaxes(x,1,2)
print(np.shape(x))
print(np.shape(y))
print(np.shape(z))

from keras import regularizers
from tensorflow.keras.metrics import Recall,Precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,Activation,Conv1D,MaxPooling1D,Dropout

#callbacks = myCallback()

model = Sequential()

model.add(Flatten(input_shape=(2048,)))

model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy',Recall(),Precision()])


model.fit(z, y, epochs=90, batch_size=128, validation_split=0.1, callbacks=[es,csv,checkpoint])


model.evaluate(z, y, verbose=2)

model.summary()

y1=model.predict(z)
c=tf.math.confusion_matrix(y,y1)
with tf.Session() as sess:
    confusion=sess.run(c)
print(confusion)

model.save('diff_model.h5')

import pandas as pd
df=pd.read_csv("metrics.csv")
print(df)


model.load_weights("weights.12-0.55.hdf5")

model.evaluate(z, y, verbose=1)

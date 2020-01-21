import tensorflow as tf
import numpy as np
import pickle
import cv2 
import matplotlib.pyplot

new_model = tf.keras.models.load_model('diff_model.h5')
new_model.summary()

encode_model = tf.keras.models.load_model('encode_model.h5')
encode_model.summary()

data=pickle.load(open('one_shot_inputs.pickle','rb'))
print(np.shape(data))

data=np.asarray(data)
print(data)

label=np.array(data[:,1])
print(label)
print(np.shape(label))
print(np.shape(data))

image_path="../Desktop/test_2.jpg"
imgs=cv2.imread(image_path,cv2.IMREAD_COLOR)
new=cv2.resize(imgs,(299,299))
input_data=encode_model.predict(np.expand_dims(new,0))
print(np.shape(input_data))

x=np.zeros((200,2048,))
i=0
for y in data[:,0]:
    x[i,:]=np.sqrt(np.square(y-input_data[0]))
    i=i+1
print(np.shape(x))
print(x)

y=new_model.predict(x)

n=np.argmax(y)
print(n)
print(label[n])

if y[n]>0.5:
    print(label[n])

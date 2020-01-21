import pickle
import cv2
import os
import numpy as np
from keras.applications.inception_v3 import InceptionV3
encode_model=InceptionV3(weights='imagenet')
encode_model.layers.pop()
for layer in encode_model.layers:
    layer.trainable=False
print(encode_model.summary())
from keras import Model
model_output = encode_model.get_layer("avg_pool").output
model = Model(inputs=encode_model.input, outputs=model_output)
f = '../img'
x={}
for filename in os.listdir(f):
    i=0
    full_path = os.path.join(f, filename)
    y=[]
    if full_path=="../img/.ipynb_checkpoints":
        continue
    if os.path.isdir(full_path):
        for fg in os.listdir(full_path):
            im = os.path.join(full_path, fg)
            if im.endswith(".jpg"):
                imgs=cv2.imread(im,cv2.IMREAD_COLOR)
                new=cv2.resize(imgs,(299,299))
                a=model.predict(np.expand_dims(new,0))
                y.append([a])
        y=np.reshape(y,(np.shape(y)[0],np.shape(y)[3]))
        x[filename]=y
print(len(x))
print(np.shape(x['ic']))
print(x['ic'])
print(len(x['ic']))
print(len(x))
n=len(x['ic'])
l=len(x)
print(n)
print(l)

m=0
for y in x.values():
    m=m+np.shape(y)[0]
print(m)
size=m*n*2
print(m*n*2)

y1=np.empty((size,),dtype=np.int_)
i=0
for (label,y) in x.items():
    for a in y:
        data[i:i+n,0,:]=a
        data[i:i+n,1,:]=y
        y1[i:i+n]=1
        i=i+n
        
j=0
for (label,y) in x.items():
    k=['ab3','ic','library','shops']
    del k[k.index(label)]
    for a in y:
        data[i:i+n,0,:]=a
        data[i:i+n,1,:]=x[k[j]]
        y1[i:i+n]=0
        i=i+n
        j=(j+1)%(l-1)
        
print(i)
z=np.sqrt(np.square(np.subtract(data[:,0,:],data[:,1,:])))
data[:,2,:]=z
print(z)
print(np.shape(z))

import random
random.shuffle(new)

data=new[1]
y1=new[0]
print(np.shape(data))
print(np.shape(y1))

new=[]
for (label,y) in x.items():
    for a in y:
        new.append([a,label])
print(np.shape(new[0][0]))
print(np.shape(new))
print(new[0][1])



print(new)

pickle_out=open('one_shot_pre_data.pickle','wb')
pickle.dump(data,pickle_out)
pickle_out.close()

pickle_out=open('one_shot_pre_label.pickle','wb')
pickle.dump(y1,pickle_out)
pickle_out.close()
model.save('encode_model.h5')

pickle_out=open('one_shot_pre_data_2.pickle','wb')
pickle.dump(z,pickle_out)
pickle_out.close()

pickle_out=open('one_shot_inputs.pickle','wb')
pickle.dump(new,pickle_out)
pickle_out.close()

import numpy as np
import cv2
import h5py
import scipy.io
from imgaug import augmenters as iaa
import pandas as pd
import gc

from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
from tensorflow.keras.activations import *

lbl=[]
img=np.zeros((3064,224,224))
for i in range(1,3065):
    try:
        path='/kaggle/input/brain-tumour/brainTumorDataPublic_1766/'
        with h5py.File(path+str(i)+'.mat') as f:
          images = f['cjdata']
          resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )
          x=np.asarray(resized)
          x=(x-np.min(x))/(np.max(x)-np.min(x))
          x=x.reshape((1,224,224))
          img[i-1]=x
          lbl.append(int(images['label'][0]))
    except:
        try:
          path='/kaggle/input/brain-tumour/brainTumorDataPublic_22993064/'
          with h5py.File(path+str(i)+'.mat') as f:
              images = f['cjdata']
              resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )
              x=np.asarray(resized)
              x=(x-np.min(x))/(np.max(x)-np.min(x))
              x=x.reshape((1,224,224))
              img[i-1]=x
              lbl.append(int(images['label'][0]))
        except:
            try:
              path='/kaggle/input/brain-tumour/brainTumorDataPublic_15332298/'
              with h5py.File(path+str(i)+'.mat') as f:
                  images = f['cjdata']
                  resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )
                  x=np.asarray(resized)
                  x=(x-np.min(x))/(np.max(x)-np.min(x))
                  x=x.reshape((1,224,224))
                  img[i-1]=x
                  lbl.append(int(images['label'][0]))
            except:
              path='/kaggle/input/brain-tumour/brainTumorDataPublic_7671532/'
              with h5py.File(path+str(i)+'.mat') as f:
                  images = f['cjdata']
                  resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )
                  x=np.asarray(resized)
                  x=(x-np.min(x))/(np.max(x)-np.min(x))
                  x=x.reshape((1,224,224))
                  img[i-1]=x
                  lbl.append(int(images['label'][0]))

path='/kaggle/input/braintumour/cvind.mat'

with h5py.File(path) as f:
      data=f['cvind']
      idx=data[0]
obj_arr = {}
obj_arr['images'] = img
obj_arr['label'] = lbl
obj_arr['fold']=idx


df=obj_arr
df=df

def Global_attention_block(inputs):
    shape=K.int_shape(inputs)
    x=Lambda(lambda x: K.mean(x,-1))(inputs)
    x=Reshape((1,-1))(x)
    x=sigmoid(x)# 1,HW
    
    y=Reshape((-1,shape[-1]))(inputs)#HW C
    
    x=K.batch_dot(x,y)#1 C
    
    p=GlobalAveragePooling2D() (inputs)
    p=Reshape((1,1,shape[-1]))(p)#1 1 C
    p=Conv2D(shape[-1],1,activation='relu')(p)
    p=Conv2D(shape[-1],1,activation='sigmoid')(p)
    p=Reshape((1,shape[-1]))(p)#1 C
    
    
    x=Concatenate()([x,p])
    x=Reshape((1,1,2*shape[-1]))(x)#1 1 C
    x=Conv2D(shape[-1],1,activation='sigmoid')(x)
    
    return Multiply()([x,inputs])


def load_model():   
  K.clear_session() 
  mod1=DenseNet121(input_shape=(224,224,3))
  mod1.trainable=False
  out_1=mod1.layers[-3].output
  p=Lambda(lambda x: x[:,:,:, :512])(out_1)
  q=Lambda(lambda x: x[:,:,:, 512:])(out_1)
    
  p = Global_attention_block(p)
  q = Global_attention_block(q)
  
  out_1=Concatenate()([p,q])
  out_1 = GlobalAveragePooling2D()(out_1)
  out=Dense(3,activation='softmax')(out_1)
  model=Model(inputs=mod1.input,outputs=out)
  return model
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
def get_trn_tst(df,tst_fold):
  idx=np.asarray(df['fold'])
  y=np.asarray(df['label'])
  y-=1
  img=np.asarray(df['images'])
  img1=[]
  for i in range(len(img)):
        img1.append(change(img[i]))
  img1=np.asarray(img1)
  del([img])
  gc.collect()
  trn_y=np.asarray(y[(idx!=tst_fold)])
  trn_img=np.asarray(img1[(idx!=tst_fold)])
  tst_y=np.asarray(y[(idx==tst_fold)])
  tst_img=img1[idx==tst_fold]
  trn_img=np.repeat(trn_img.reshape((trn_img.shape[0],224,224,1)),3,axis=3)
  tst_img=np.repeat(tst_img.reshape((tst_img.shape[0],224,224,1)),3,axis=3)
  return (trn_img.copy(),trn_y.copy()),(tst_img.copy(),tst_y.copy())

def change(img):
    resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA )
    return resized

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
def Hflip( images):
		seq = iaa.Sequential([iaa.Fliplr(1.0)])
		return seq.augment_images(images)
def Vflip( images):
		seq = iaa.Sequential([iaa.Flipud(1.0)])
		return seq.augment_images(images)
def noise(images):
    ls=[]
    for i in images:
        x = np.random.normal(loc=0, scale=0.05, size=(299,299,3))
        ls.append(i+x)
    return ls
def rotate(images):
    ls=[]
    for angle in range(-15,20,5):
        for image in images:
            ls.append(rotate_image(image,angle))
    return ls
class DataGenerator(Sequence):
  def __init__(self, images, labels, batch_size=64, image_dimensions = (96 ,96 ,3), shuffle=False, augment=False):
    self.labels       = labels              # array of labels
    self.images = images        # array of image paths
    self.batch_size   = batch_size          # batch size
    self.on_epoch_end()

  def __len__(self):
    return int(np.floor(self.labels.shape[0] / self.batch_size))

  def on_epoch_end(self):
    self.indexes = np.arange(self.labels.shape[0])

  def __getitem__(self, index):
    indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
    labels = self.labels.loc[indexes]
    img = [self.images[k].astype(np.float32) for k in indexes]
    imgH=Hflip(img)
    imgV=Vflip(img)
    imgR=rotate(img)
    images=[]
    images.extend(imgH)
    images.extend(imgV)
    images.extend(imgR)
    lbl=labels.copy()
    labels=pd.DataFrame()
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    return np.asarray(images), np.asarray(labels.values)
def upd(dk,data):
  if dk==0:
    dk=data
  else:
    for ky in data.keys():
      dk[ky].extend(data[ky])
  return dk
index=1
epoch=50
pre_acc=0
best=0
fold='fold_'+str(index)
trn,tst=get_trn_tst(df,index)
model=load_model()
trn_x,trn_y=unison_shuffled_copies(trn[0],trn[1])
tst_x,tst_y=unison_shuffled_copies(tst[0],tst[1])
train_data = DataGenerator(trn_x,pd.get_dummies(trn_y), batch_size=4, augment=True)
ln=len(trn_y)
model.compile(optimizer=Adam(1e-2,decay=1e-3), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
hist=model.fit_generator(train_data,epochs=50,verbose=1,steps_per_epoch=ln//4)
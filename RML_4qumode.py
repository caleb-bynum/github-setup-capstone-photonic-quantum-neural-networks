# %% [markdown]
# ### Author: Samuel Smith, Arthur Lobo

# %% [markdown]
# #### Portland State University, Electrical and Computer Engineering

# %% [markdown]
# Dependencies: keras-nightly==2.5.0.dev2021032900 PennyLane==0.17.0 StrawberryFields==0.18.0 tensorflow-2.4.0-cp38-cp38-macosx_10_9_x86_64.whl

# %% [markdown]
# # 4-qumode classifier
# 
# Classical and Continuous Variable Quantum hybrid network: Classical layers using keras dense and quantum layers using Pennylane

# %%
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.models as models
from tensorflow.keras.layers import Reshape,Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D 

import pennylane as qml
import numpy as np

import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.get_logger().setLevel('ERROR')

# %% [markdown]
# ## 0. Loading data

# %% [markdown]
# Normalize pixel values from 0 ~ 255 to 0 ~ 1

# %%
Xd = pickle.load(open("./RML2016.10a_dict.pkl",'rb'),encoding = "bytes")
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])

# remove 7 modulations to only train on 4 modulations
#mods.remove(b'QPSK')
mods.remove(b'8PSK')
mods.remove(b'AM-DSB')
#mods.remove(b'AM-SSB')
mods.remove(b'PAM4')
mods.remove(b'QAM16')
#mods.remove(b'QAM64')
mods.remove(b'BPSK')
mods.remove(b'CPFSK')
mods.remove(b'GFSK')
#mods.remove(b'WBFM')

# read in data
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

# Partition the data into training and test sets 
np.random.seed(2023)
n_examples = X.shape[0]
n_train = (3 * n_examples) // 4        # 75% to train

idx = np.random.choice(range(0,n_examples), size=n_examples, replace=False)
train_idx = idx[0:n_train]
test_idx = idx[n_train:n_examples]

div_factor = 10    # To use 100th of the dataset because 200,000 vectors may be too computationally expensive to train

train_idx = train_idx[0:len(train_idx)//div_factor]
test_idx = test_idx[0:len(test_idx)//div_factor]
print(len(train_idx), len(test_idx))

X_train = X[train_idx]
X_test =  X[test_idx]

#one-hot encode the labels
lb = preprocessing.LabelBinarizer()
lb.fit(np.asarray(lbl)[:,0])
print(lb.classes_)
lbl_encoded=lb.transform(np.asarray(lbl)[:,0])
y_train=lbl_encoded[train_idx]
y_test=lbl_encoded[test_idx]

in_shp = list(X_train.shape[1:])

# %% [markdown]
# One hot encode labels to vectors of size cutoff_dim^(num_qumodes)

# %%
def one_hot(labels):  
       
    depth =  2**4                       # 4 classes + 12 zeros for padding
    indices = labels.astype(np.int32)    
    one_hot_labels = np.eye(depth)[indices].astype(np.float32) 
    
    return one_hot_labels

y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

# one-hot encoded labels, each label of length cutoff dimension**2
y_train, y_test = one_hot(y_train), one_hot(y_test)

# %% [markdown]
# ## 1. Classical circuit

# %%
keras.backend.set_floatx('float64')

# Define classical layers using Keras Sequential. Take in 2x128 radio modulations, flatten, and output vectors of length 30. 2 hidden layers with ELU activation.
model = models.Sequential()
model.add(Reshape(in_shp + [1], input_shape = in_shp))

model.add(Conv2D(64, (1, 16), activation ='relu'))
model.add(Dropout(0.55))

model.add(Conv2D(32, (2, 8), activation ='relu'))
model.add(Dropout(0.55))


model.add(Conv2D(16, (1, 4), activation ='relu'))
model.add(Dropout(0.55))

model.add(Flatten())

model.add(Dense(30, activation ='sigmoid'))


# More than a million parameters for the classical circuit
model.summary()

# %% [markdown]
# 

# %% [markdown]
# ## 2. Data encoding circuit

# %% [markdown]
# Encode the output vectors from the classical network into quantum states using the vector entries as parameters of continuous variable gates. 

# %%
def encode_data(x):
    qml.Squeezing(x[0], x[1], wires=0)
    qml.Squeezing(x[2], x[3], wires=1)
    qml.Squeezing(x[4], x[5], wires=2)
    qml.Squeezing(x[6], x[7], wires=3)
    
    qml.Beamsplitter(x[8], x[9], wires=[0,1])
    qml.Beamsplitter(x[10], x[11], wires=[1,2])
    qml.Beamsplitter(x[12], x[13], wires=[2,3])
    
    qml.Rotation(x[14], wires=0)
    qml.Rotation(x[15], wires=1)
    qml.Rotation(x[16], wires=2)
    qml.Rotation(x[17], wires=3)    
    
    qml.Displacement(x[18], x[19], wires=0)
    qml.Displacement(x[20], x[21], wires=1)
    qml.Displacement(x[22], x[23], wires=2)
    qml.Displacement(x[24], x[25], wires=3) 
    
    qml.Kerr(x[26], wires=0)
    qml.Kerr(x[27], wires=1)
    qml.Kerr(x[28], wires=2)
    qml.Kerr(x[29], wires=3)

# %% [markdown]
# ## 3. Qauntum neural network circuit

# %%
def layer(v):
    
    # Linear transformation W = Interferemeter, squeezers, interferometer
    # Interferometer 1
    qml.Beamsplitter(v[0], v[1], wires=[0,1])
    qml.Beamsplitter(v[2], v[3], wires=[1,2])
    qml.Beamsplitter(v[4], v[5], wires=[2,3])
    
    qml.Rotation(v[6], wires=0)
    qml.Rotation(v[7], wires=1)
    qml.Rotation(v[8], wires=2)
    qml.Rotation(v[9], wires=3)
    
    # Squeezers
    qml.Squeezing(v[10], v[11], wires=0)
    qml.Squeezing(v[12], v[13], wires=1)
    qml.Squeezing(v[14], v[15], wires=2)
    qml.Squeezing(v[16], v[17], wires=3) 
    
    # Interferometer 2
    qml.Beamsplitter(v[18], v[19], wires=[0,1])
    qml.Beamsplitter(v[20], v[21], wires=[1,2])
    qml.Beamsplitter(v[22], v[23], wires=[2,3])
    
    qml.Rotation(v[24], wires=0)
    qml.Rotation(v[25], wires=1)
    qml.Rotation(v[26], wires=2)
    qml.Rotation(v[27], wires=3)
    
    # Bias addition
    qml.Displacement(v[28], v[29], wires=0)
    qml.Displacement(v[30], v[31], wires=1)
    qml.Displacement(v[32], v[33], wires=2)
    qml.Displacement(v[34], v[35], wires=3)
    
    # Non-linear activation
    qml.Kerr(v[36], wires=0)
    qml.Kerr(v[37], wires=1)
    qml.Kerr(v[38], wires=2)
    qml.Kerr(v[39], wires=3)

# %% [markdown]
# ## 4. Quantum device

# %% [markdown]
# For the expression of qumodes in Fock basis, choose a "strawberryfields.fock" device. Define the number of qumodes and cutoff dimension. Run the data encoding circuit and quantum neural network circuit. The probability measurement method (qml.probs(wires)) returns vectors of size 2^4 = 16 (cutoff_dim^num_modes).

# %%
num_modes = 4
cutoff_dim = 2

# select a devide 
dev = qml.device("strawberryfields.fock", wires=num_modes, cutoff_dim=cutoff_dim) 

@qml.qnode(dev, interface="tf")
def quantum_nn(inputs, var):
    # Encode input x into quantum state
    encode_data(inputs)

    # iterative quantum layers
    for v in var:
        layer(v)

    # Encode input x into quantum state
    encode_data(inputs)

    # iterative quantum layers
    for v in var:
        layer(v)

    return qml.probs(wires=[0, 1, 2, 3])  # Measurement

# %% [markdown]
# ## 5. Hybrid circuit

# %%
num_layers = 4
weight_shape = {'var': (num_layers, 40)}          # 4 layers and 40 parameters per layer, Keras layer will initialize.

qlayer = qml.qnn.KerasLayer(quantum_nn, weight_shape, output_dim = 4)

# add to the classical sequential model
model.add(qlayer)

# %% [markdown]
# ## 6. Loss function and optimizer

# %%
#opt = keras.optimizers.SGD(lr = 0.02)
opt = keras.optimizers.Adam(learning_rate = 0.02)
model.compile(opt, loss = 'categorical_crossentropy', metrics =['accuracy'])
class ModelCheckpoint(keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.best_accuracy = 0.0000  # Initial value is 0, if warm-starting use value of the best validation accuracy so far
        self.f1 = open("loss_accuracy_CV_4_qumodes_co4", 'a')   # change name for your specific case

    def on_epoch_end(self, epoch, logs=None):
#        print(self.model.get_layer('dense').get_weights())
#        print(self.model.get_layer('keras_layer').get_weights())
        self.f1.write("%f %f %f %f\n" % (logs['loss'], logs['accuracy'], logs['val_loss'], logs['val_accuracy']))
        self.f1.flush()
        if logs['val_accuracy'] > self.best_accuracy:
            print("saving weights")
            self.best_accuracy = logs['val_accuracy']
            self.model.save_weights('RML_CV_4_qumodes_co4')   # change name for your specific case

ckpt = ModelCheckpoint()


#Uncomment the following two lines for warm start
#model.load_weights('RML_CV_4_qumodes_co2')
#print('loaded model')

# %% [markdown]
# ## 7. Training

# %%
hybrid = model.fit(X_train, 
                   y_train,
                   epochs = 150,
                   batch_size = 256,
                   shuffle = True, 
                   validation_data = (X_test, y_test),
                   callbacks = [ckpt])

# %%
model.summary()

# %% [markdown]
# ## 8. Loss and accuracy graphs

# %%
# ===================================================================================
#                                  Loss History Plot
# ===================================================================================

plt.title('model loss')
plt.plot(hybrid.history['loss'], '-g')
plt.ylabel('loss')
plt.show()

# %%
# ===================================================================================
#                                Accuracy History Plot
# ===================================================================================

plt.title('model accuracy')
plt.plot(hybrid.history['accuracy'], '-g')
plt.ylabel('accuracy')
plt.show()



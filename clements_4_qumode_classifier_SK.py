# %% [markdown]
# ### Original Author: Sophie Choe
# ### Updates by: Caleb Bynum [Texas A&M], Samuel Smith [Texas A&M]

# %% [markdown]
# #### Portland State University, Electrical and Computer Engineering

# %% [markdown]
# 

# %% [markdown]
# Dependencies: keras-nightly==2.5.0.dev2021032900 PennyLane==0.17.0 StrawberryFields==0.18.0 tensorflow-2.4.0-cp38-cp38-macosx_10_9_x86_64.whl

# %%


# %% [markdown]
# # 4-qumode classifier implemented with Reck Mesh.
# 
# Classical and Continuous Variable Quantum hybrid network: Classical layers using keras dense and quantum layers using Pennylane

# %%
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

import pennylane as qml
import numpy as np

import matplotlib.pyplot as plt
import os
#s.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# %% [markdown]
# ## 0. Loading data

# %% [markdown]
# Normalize pixel values from 0 ~ 255 to 0 ~ 1

# %%
mnist = keras.datasets.mnist

# datasets are numpy.ndarrays
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()     

# normalize the image data
X_train, X_test = X_train / 255.0, X_test / 255.0

# %% [markdown]
# One hot encode labels to vectors of size cutoff_dim^(num_qumodes)

# %%
def one_hot(labels):  
       
    depth =  2**4                       # 10 classes + 6 zeros for padding
    indices = labels.astype(np.int32)    
    one_hot_labels = np.eye(depth)[indices].astype(np.float32) 
    
    return one_hot_labels

# one-hot encoded labels, each label of length cutoff dimension**2
y_train, y_test = one_hot(Y_train), one_hot(Y_test)

# using only 600 samples for training in this experiment
n_samples = 600
test_samples = 100
X_train, X_test, y_train, y_test = X_train[:n_samples], X_test[:test_samples], y_train[:n_samples], y_test[:test_samples]

# %% [markdown]
# ## 1. Classical circuit

# %% [markdown]
# Define classical layers using Keras Sequential. Take in 28x28 image matrices, flatten, and output vectors of length 30. 2 hidden layers with ELU activation.

# %%
keras.backend.set_floatx('float64')

model = keras.models.Sequential([
                                 layers.Flatten(input_shape = (28,28)),
                                 layers.Dense(392, activation ="elu"),
                                 layers.Dense(196, activation ="elu"),
                                 layers.Dense(98),
                                 layers.Dense(49, activation ="elu"),
                                 layers.Dense(36), # clem mesh encode_data param count
                                ])

# More than a million parameters for the classical circuit
model.summary()


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
    qml.Beamsplitter(x[10], x[11], wires=[2,3])

    qml.Beamsplitter(x[12], x[13], wires=[1,2])

    qml.Beamsplitter(x[14], x[15], wires=[0,1])
    qml.Beamsplitter(x[16], x[17], wires=[2,3])

    qml.Rotation(x[18], wires=0)
    qml.Beamsplitter(x[19], x[20], wires=[1,2])
    qml.Rotation(x[21], wires=3)

    qml.Displacement(x[22], x[23], wires=0)
    qml.Rotation(x[24], wires=1)
    qml.Rotation(x[25], wires=2)
    qml.Displacement(x[26], x[27], wires=3)

    qml.Kerr(x[28], wires=0)
    qml.Displacement(x[29], x[30], wires=1)
    qml.Displacement(x[31], x[32], wires=2)
    qml.Kerr(x[33], wires=3)

    qml.Kerr(x[34], wires=1)
    qml.Kerr(x[35], wires=2)



# %% [markdown]
# ## 3. Qauntum neural network circuit

# %%
def layer(v):
    qml.Beamsplitter(v[0], v[1], wires=[0,1])
    qml.Beamsplitter(v[2], v[3], wires=[2,3])

    qml.Beamsplitter(v[4], v[5], wires=[1,2])

    qml.Beamsplitter(v[6], v[7], wires=[0,1])
    qml.Beamsplitter(v[8], v[9], wires=[2,3])

    qml.Rotation(v[10], wires=0)
    qml.Beamsplitter(v[11], v[12], wires=[1,2])
    qml.Rotation(v[13], wires=3)

    qml.Squeezing(v[14], v[15], wires=0)
    qml.Rotation(v[16], wires=1)
    qml.Rotation(v[17], wires=2)
    qml.Squeezing(v[18], v[19], wires=3)

    qml.Kerr(v[20], wires=0)
    qml.Squeezing(v[21], v[22], wires=1)
    qml.Squeezing(v[23], v[24], wires=2)
    qml.Kerr(v[25], wires=3)

    qml.Kerr(v[26], wires=1)
    qml.Kerr(v[27], wires=2)

    qml.Beamsplitter(v[28], v[29], wires=[0,1])
    qml.Beamsplitter(v[30], v[31], wires=[2,3])

    qml.Beamsplitter(v[32], v[33], wires=[1,2])

    qml.Beamsplitter(v[34], v[35], wires=[0,1])
    qml.Beamsplitter(v[36], v[37], wires=[2,3])

    qml.Rotation(v[38], wires=0)
    qml.Beamsplitter(v[39], v[40], wires=[1,2])
    qml.Rotation(v[41], wires=3)

    qml.Displacement(v[42], v[43], wires=0)
    qml.Rotation(v[44], wires=1)
    qml.Rotation(v[45], wires=2)
    qml.Displacement(v[46], v[47], wires=3)

    qml.Kerr(v[48], wires=0)
    qml.Displacement(v[49], v[50], wires=1)
    qml.Displacement(v[51], v[52], wires=2)
    qml.Kerr(v[53], wires=3)

    qml.Kerr(v[54], wires=2)
    qml.Kerr(v[55], wires=3)




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

    return qml.probs(wires=[0, 1, 2, 3])  # Measurement

# %% [markdown]
# ## 5. Hybrid circuit

# %%
weight_shape = {'var': (4,56)}          # 4 layers and 56 parameters per layer, Keras layer will initialize.
# num element, size 
num_layers = 4
qlayer = qml.qnn.KerasLayer(quantum_nn, weight_shape, output_dim = 4)

# add to the classical sequential model
model.add(qlayer)


# %% [markdown]
# ## 6. Loss function and optimizer

# %%
opt = keras.optimizers.SGD(lr = 0.02)
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
            self.model.save_weights('CV_2_qumodes_co4')   # change name for your specific case

ckpt = ModelCheckpoint()


# %% [markdown]
# ## 7. Training

# %%
hybrid = model.fit(X_train, 
                   y_train,
                   epochs = 70,
                   batch_size = 64,
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



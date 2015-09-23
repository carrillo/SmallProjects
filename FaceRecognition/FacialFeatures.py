__author__ = 'carrillo'

# Imports for loading
import os

import pandas as pd
import numpy as np

# Imports for neural net
from lasagne import layers
from nolearn.lasagne import NeuralNet
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import rectify
from lasagne.objectives import mse
from flipBatchIterator import FlipBatchIterator
from updateParameter import UpdateParameter
import theano
# Imports for plotting
from matplotlib import pyplot

# Imports for serialization
import cPickle as pickle
import sys
sys.setrecursionlimit(10000)

def load(file, drop_nas=True, targets_keep = None ):
    """
    Load file and split into X and y for training set.
    1. Load file.
    2. Transform space separated pixel values to numpy array
    3. Keep only specified targets. Keep all if none specified.
    4./5. Split X and Y data and scale such that pixel values scale from 0 to 1 and coordinates from -1 to 1.
    """

    print('Loading data.')
    # 1. Load file, shuffle and test if train or test set.
    XY = pd.read_csv(os.path.expanduser(file))
    XY = XY.reindex(np.random.permutation(XY.index))
    XY = XY.dropna()
    train = XY.shape[1] > 2


    # 2. Transform pixel values to numpy array.
    # Use anonymous function to call np.fromstring(XX, sep=" ") on image string
    XY['Image'] = XY['Image'].apply(lambda pixel_string: np.fromstring(pixel_string, dtype=np.float32, sep=' ') )

    # 3. Keep only specified target values.
    if targets_keep and train: XY = XY[targets_keep + ['Image']]

    # 4. Extract image values as a stack of numpy arrays and scale pixel values from 0-255 to 0-1.
    X = np.vstack(XY['Image'].values / 255)
    Y = None

    # 5. Process target values if train data.
    if train:
        Y = XY.drop('Image', axis=1).values # Remove 'Image' columns
        Y = ( Y - 48 ) / 48 # Convert from 96x96 pixel dimension to -1 to 1 scale.
        Y = Y.astype(np.float32)

    print('Loading data. Done')

    return X, Y

# Wraps the load function to represent each input as an 1x96x96 array to keep local correlations exploited by convolution.
def load2d(file, drop_nas=True, targets_keep = None ):
    X, Y = load(file, drop_nas=drop_nas, targets_keep=targets_keep )
    X = X.reshape(-1,1,96,96) # Infer the first dimension from the length of the input array (samples): -1, define rest as 1,96,96
    return  X, Y


# Simple neural network with 100 hidden layers and one output unit for all 30 target values.
oneLayerNet = NeuralNet(
    layers=[    # Simple neural net with one hidden layer.
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
    ],

    # Network architecture
    input_shape=(None, 9216), # 96x96 pixels for input sample. None for non-specified sample-batch size.
    hidden_nonlinearity=rectify, # Set rectified linear function as activation funciton of hidden layer.
    hidden_num_units=100, # 100 hidden units
    output_nonlinearity=None, # no transformation of output. Use identity function
    output_num_units=30, # One output unit for each target value


    # Parameter optimization. Use Nesterov Momentum with learning rate 0.01 and momentum 0.9
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    # General setting
    objective_loss_function=mse,
    regression=True,
    max_epochs=400,
    verbose=1,
    eval_size=0.2,
    )

def float32(x):
    return np.cast['float32'](x)

# Simple neural network with 100 hidden layers and one output unit for all 30 target values.
testNet = NeuralNet(
    layers=[    # Simple neural net with one hidden layer.
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
    ],

    # Network architecture
    input_shape=(None, 1, 96, 96), # 96x96 pixels for input sample. None for non-specified sample-batch size.
    hidden_nonlinearity=rectify, # Set rectified linear function as activation funciton of hidden layer.
    hidden_num_units=1000, # 100 hidden units
    output_nonlinearity=None, # no transformation of output. Use identity function
    output_num_units=30, # One output unit for each target value


    # Parameter optimization. Use Nesterov Momentum with learning rate 0.01 and momentum 0.9
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.01)),
    update_momentum=theano.shared(float32(0.9)),

    # General setting
    objective_loss_function=mse,
    #batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        UpdateParameter('update_learning_rate', start=0.03, stop=0.0001),
        UpdateParameter('update_momentum', start=0.9, stop=0.999),
        ],
    regression=True,
    max_epochs=400,
    verbose=1,
    eval_size=0.2,
    )


convolutionNetwork = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
    ],

    # Network architecture
    input_shape=(None, 1, 96, 96), # 96x96 pixels for input sample. None for non-specified sample-batch size.
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
    hidden4_nonlinearity=rectify, hidden4_num_units=500,
    hidden5_nonlinearity=rectify, hidden5_num_units=500,
    output_nonlinearity=None, output_num_units=30,

    # Parameter optimization. Use Nesterov Momentum with learning rate 0.01 and momentum 0.9
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    # General setting
    objective_loss_function=mse,
    regression=True,
    max_epochs=3000,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        UpdateParameter('update_learning_rate', start=0.03, stop=0.0001),
        UpdateParameter('update_momentum', start=0.9, stop=0.999),
        ],
    verbose=1,
    eval_size=0.2,
    )

def float32(x):
    return np.cast['float32'](x)

convolutionNetwork_learningUpdate = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
    ],

    # Network architecture
    input_shape=(None, 1, 96, 96), # 96x96 pixels for input sample. None for non-specified sample-batch size.
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
    hidden4_nonlinearity=rectify, hidden4_num_units=500,
    hidden5_nonlinearity=rectify, hidden5_num_units=500,
    output_nonlinearity=None, output_num_units=30,

    # Parameter optimization. Use Nesterov Momentum
    # Specify learning_rate and momentum as theano shared variable. This allows updating during learning.
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    # General setting
    objective_loss_function=mse,
    regression=True,
    max_epochs=3000,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    verbose=1,
    eval_size=0.2,
    )

convolutionNetwork_dropout = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
    ],

    # Network architecture
    input_shape=(None, 1, 96, 96), # 96x96 pixels for input sample. None for non-specified sample-batch size.
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2), dropout1_p=0.1,
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2), dropout2_p=0.2,
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2), dropout3_p=0.3,
    hidden4_nonlinearity=rectify, hidden4_num_units=500, dropout4_p=0.5,
    hidden5_nonlinearity=rectify, hidden5_num_units=500,
    output_nonlinearity=None, output_num_units=30,

    # Parameter optimization. Use Nesterov Momentum
    # Specify learning_rate and momentum as theano shared variable. This allows updating during learning.
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    # General setting
    objective_loss_function=mse,
    regression=True,
    max_epochs=3000,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    verbose=1,
    eval_size=0.2,
    )


def plot_loss(neural_net):
    train_loss = np.array([epoch['train_loss'] for epoch in neural_net.train_history_])
    test_loss = np.array([epoch['valid_loss'] for epoch in neural_net.train_history_])
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(test_loss, linewidth=3, label="test")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    #pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")
    pyplot.show()

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

def plot_samples(x, y):
    """
    Plots 16 samples of X and Y set.
    :param x: Array of Images (2d)
    :param y: Array of targets
    :return:
    """
    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(x[i], y[i], ax)
    pyplot.show()

def mirror_vertical(x, y):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    X_flipped = x[:,:,:,::-1]

    if y is not None:
        # Flip x coordinates. X coordinates are every second value.
        # Coordinates are symmetrical around 0.
        # Flipping is performed by multiplication by -1
        Y_flipped = y #np.copy(y)
        Y_flipped[:,::2] = Y_flipped[:,::2]*-1
        

    return X_flipped, Y_flipped


train_file = '~/workspace/SmallProjects/FaceRecognition/data/training.csv'
test_file = "~/workspace/SmallProjects/FaceRecognition/data/test.csv"

#X_train, Y_train = load(train_file)
#X_test, _ = load(test_file)

#oneLayerNet.fit(X_train, Y_train)
#plot_loss(oneLayerNet)
#y_predict = oneLayerNet.predict(X_test)
#plot_samples(X_test, y_predict)
#with open('simpleNet.pickle', 'wb') as f:
    #pickle.dump(oneLayerNet, f, -1)



X_train, Y_train = load2d(train_file)
X_test, _ = load2d(test_file)

#testNet.fit(X_train, Y_train)
#plot_loss(testNet)
#y_predict = testNet.predict(X_test)
#plot_samples(X_test, y_predict)


#convolutionNetwork.fit(X_train, Y_train)
#with open('convNetFlippedSamples.pickle', 'wb') as f:
    #pickle.dump(convolutionNetwork,f , -1)

convolutionNetwork_dropout.fit(X_train, Y_train)
with open('convNetDropout.pickle', 'wb') as f:
    pickle.dump(convolutionNetwork_learningUpdate, f, -1)

#X_train_flipped = X_train[:, :, :, ::-1]  # simple slice to flip all images
#fig = pyplot.figure(figsize=(6, 3))
#ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])

#plot_sample(X_train[1], Y_train[1], ax)
#ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])

#X_flipped, Y_flipped = mirror_vertical(X_train, Y_train)
#plot_sample(X_flipped[1], Y_flipped[1], ax)
#pyplot.show()

#convolutionNetwork = pickle.load( open( "convNet.pickle", "rb" ) )

#plot_loss(convolutionNetwork)
#y_predict = convolutionNetwork.predict(X_test)
#plot_samples(X_test, y_predict)

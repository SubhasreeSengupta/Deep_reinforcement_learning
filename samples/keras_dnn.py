from keras.preprocessing import image
from keras.models import Model
import numpy
import sklearn.metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

def keras_dnn(meanFile, trainFile, testFile, labelColumn=1, maxCount=255.,
              classes=[0,1,2]):
    """
    title::
        keras_dnn

    description::
        Implementation of a (deep) neural network using Keras FOR 
        CLASSIFICATION. Change the configuration of the network and other 
        parameters to improve results. Input data should be sorted into 
        training/test already.

    attributes::
        meanFile
            file containing the mean of the entire dataset (only necessary for
            image data, just delete this for other data)
        trainFile
            file containing training samples in rows, with the labelColumn 
            containing the labels for that row of data

            Ex: [[1 1 1 1 1 0]
                 [0 1 0 1 0 2]
                 [0 0 0 0 0 1]] 

            (so there are 5 input features, classes 0,1,2, labelColumn = 1)

        testFile
            file containing test samples in rows, with the labelColumn
            containing the labels  for that row of data (same format as 
            trainFile)
        labelColumn
            column containing label data (start counting by 1, from the right
            side of the array; the last column has labelColumn = 1, so when 
            indexing, we can do [:,:-labelColumn]; this was added because
            the label column in my data is 5, and the last 4 columns contain
            video, frame, grid, and overall indices to each row of data)
            [default is 1, so last column is default location for labeled data]
        maxCount
            maximum digital count for imagery [default is 255. for 8-bit images]
        classes
            class labels, should be list of integers corresponding to 
            class labels in labelColumn [default is [0,1,2], for 3 classes]

            NOTE: if you want to provide labels in the format 
            [[1,0,0],[0,1,0],[0,0,1]], you'll have to change the indexing
            slightly, you'll have to change 
            loss='sparse_categorical_crossentropy' to 
            loss='categorical_crossentropy', and you'll have to change this
            (classes) a bit for sklearn (I'd suggest sticking with [0,1,2,...])

    author::
        Elizabeth Bondi 
    """
    #Read in test/training data file, comma delimited.
    training = numpy.loadtxt(trainFile, delimiter=',')
    test = numpy.loadtxt(testFile, delimiter=',')

    #Normalization 0-1. For images: subtract the mean, normalize from 0 to 1.
    #NOTE: You may have to change this for your application!! 
    mean = numpy.loadtxt(meanFile)
    mostMin = numpy.abs(0. - mean)
    mostMax = (maxCount - mean) + mostMin
    training[:,:-labelColumn] = \
        ((training[:,:-labelColumn].astype(numpy.float64) - mean) + mostMin) / \
        mostMax
    test[:,:-labelColumn] = \
        ((test[:,:-labelColumn].astype(numpy.float64) - mean) + mostMin) / \
        mostMax

    #Example normalization for non-images (may want to get max of entire 
    #dataset, in which case I'd suggest using meanFile as maxFile):
    #training[:,:-labelColumn] = \
    #    training[:,:-labelColumn].astype(numpy.float64) / \
    #    training[:,:-labelColumn].max()
    #test[:,:-labelColumn] = test[:,:-labelColumn].astype(numpy.float64) /
    #    test[:,:-labelColumn].max()

    #Determine number of features and number of classes automatically.
    numFeatures = training.shape[1] - labelColumn 
    numClasses = numpy.unique(training[:,-labelColumn]).shape[0] 

    #Set up the model. Dense(50): fully-connected layer with 50 hidden units.
    #In the first layer, you must specify the expected input data shape:
    #for example, (# pixels)-dimensional vectors.
    
    #Change the activation functions, number of hidden layers/nodes in hidden 
    #layers, and dropout (i.e., comment it out) to try to improve results.
    #Another popular activation function is 'sigmoid', which is better for
    #smaller networks. Use 'relu' for big networks, as it converges faster.
    #’uniform’ – randomly initialize layer from uniform distribution

    nn_model=Sequential()
    nn_model.add(Dense(50, input_dim=numFeatures, init='uniform'))
    nn_model.add(Activation('relu'))
    nn_model.add(Dropout(0.5))
    nn_model.add(Dense(50, init='uniform'))
    nn_model.add(Activation('relu'))
    nn_model.add(Dropout(0.5))
    nn_model.add(Dense(numClasses, init='uniform'))
    nn_model.add(Activation('softmax'))

    #Set up objective function, and compile. 
    #learning rate = how much to change the old weight
    #decay = changes learning rate after each iteration (batch)
    #momentum = update "velocity" instead of "position" during gradient descent
    #nesterov = evaluate the gradient at the "looked-ahead" position 
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    #NOTE: use sparse_categorical_entropy instead of categorical_entropy because
    #it uses one integer target per training example (i.e., class labels are
    #[0,1,2] for sparse_c_e instead of [[1,0,0],[0,1,0],[0,0,1]] for c_e.  
    #See keras documentation for more info (https://keras.io/objectives/).
    #Use metrics=['accuracy'] for classification tasks.
    nn_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])


    #Train the neural network. Increase the number of epochs to improve
    #performance. 
    #Batch size = number of samples propagated through network at one time
    #(batch_size=10 takes the first 10 samples, trains the network, then 
    #takes the next 10 samples, trains the network, etc. to use less memory
    #during training, and to train faster)
    #Epoch = one time through ALL training data (ALL batches)
    print('fit model')
    nn_model.fit(training[:,:-labelColumn], training[:,-labelColumn],
          nb_epoch=5,
          batch_size=10)

    #Output the results for the test set. Output of .predict is of the form
    #[[1,0,0],[0,1,0],[0,0,1]], so use argmax to determine which class has the
    #maximum probability.
    import time
    start = time.time()
    preds = nn_model.predict(test[:,:-labelColumn])
    print('elapsed time [seconds]', time.time() - start)
    print('test shape', test[:,:-labelColumn].shape)
    y_pred = numpy.argmax(preds, axis=1)

    p = sklearn.metrics.precision_score(test[:,-labelColumn], y_pred,\
                                        labels=classes, average=None)
    r = sklearn.metrics.recall_score(test[:,-labelColumn], y_pred, \
                                     labels=classes, average=None)
    c = sklearn.metrics.confusion_matrix(test[:,-labelColumn], y_pred, \
                                     labels=classes)
    print('Precision: ', p)
    print('Recall: ', r)
    print('Confusion Matrix: \n', c)

if __name__ == '__main__':

    #For my data only, change this all to yours! (labelColumn probably = 1)
    trainFile = '/Users/elizabethbondi/src/python/modules/uav_id/data/60x60/1/train_class_balanced.txt'
    testFile = '/Users/elizabethbondi/src/python/modules/uav_id/data/60x60/1/test.txt'
    meanFile = '/Users/elizabethbondi/src/python/modules/uav_id/data/60x60/1/mean.txt'

    keras_dnn(meanFile, trainFile, testFile, labelColumn=5)

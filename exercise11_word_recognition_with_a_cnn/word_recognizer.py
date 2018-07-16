'''
A simple word recognizer using a CNN
built with Keras.

The trick here is to reshape the audio
data into a 2D image and let the CNN
learn features of such 2D imagees that help
to classify the audio samples.

Idea and code snippets based on the blog post of 
Manash Kumar Mandal,
https://blog.manash.me/building-a-dead-simple\
-word-recognition-engine-using-convnet-\
in-keras-25e72c19c12b

---
by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org
'''

import numpy as np
import librosa
import os
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

TRAIN_DATA_PATH = "V:/01_job/12_datasets/kaggle/word_recognition/data_train_bed_cat_happy/"
TEST_DATA_PATH  = "V:/01_job/12_datasets/kaggle/word_recognition/data_test_bed_cat_happy/"


'''
Converts a .wav file into a MFCC feature vector
(MFCC = Mel Frequency Cepstral Coefficients)
'''
def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    #print("wave shape before: ", wave.shape)
    # only use each 3rd sample
    wave = wave[::3]
    #print("wave shape after: ", wave.shape)
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc


'''
Based on the sub-dirs in the data path (as 'cat', 'happy', 'bed),
extract
 - these label names: 'cat', 'happy', 'bed'
 - get a NumPy array of label_indices: [0, 1, 2]
 - prepare a matrix of one-hot encoding vectors
# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
'''
def get_labels(path=TRAIN_DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))

    '''
    to_categorical():
    y_train = [1, 0, 3, 4, 5, 0, 2, 1]

    Assuming the labeled dataset has total six classes (0 to 5), y_train is the true label array

    np_utils.to_categorical(y_train, nb_classes=6)    
    array([[0., 1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 1.],
           [1., 0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0.]])
    '''
    return labels, label_indices, np_utils.to_categorical(label_indices)


'''
Compute for each .wav file in one sub-dir (e.g. 'cat')
the MFCC feature vector
and store all these feature vectors in a .npy data file
'''
def prepare_npy_files(path=TRAIN_DATA_PATH, max_pad_len=11):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)


'''
Get a train and a test dataset 
'''
def get_train_test(split_ratio=0.6, random_state=42):

    # get available labels
    labels, indices, _ = get_labels(TRAIN_DATA_PATH)

    # reload arrays of MFC feature vectors from disk
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # append all the datasets into one single array,
    # same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y,
                            test_size= (1 - split_ratio),
                            random_state=random_state,
                            shuffle=True)

# This needs to be called only once in order
# to generate the .npy files: cat.npy, bed.npy, happy.npy
# that contain the precomputed MFCC feature vectors
#prepare_npy_files()

# 1. get training and test data
X_train, X_test, y_train, y_test = get_train_test()
print("Shapes:")
print("X_train",X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)


# 2. reshape train and test data to 4D arrays
#    note: the CNN expects 4D tensors as input for the
#          convolutional layers!
X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)

y_train_hot = np_utils.to_categorical(y_train)
y_test_hot = np_utils.to_categorical(y_test)


# 3. setup a CNN
model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=(20, 11, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(40, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 4. train the CNN
model.fit(X_train,
          y_train_hot,
          batch_size=100,
          epochs=50,
          verbose=1,
          validation_data=(X_test, y_test_hot))

# 5. test how good the CNN works on audio streams
#    not used in the train phase
print("\nTesting the model:")
labels, _, _ = get_labels(TEST_DATA_PATH)
print(labels)
counter_correctly_predicted, counter_total = 0,0
for label in labels:
    wavfiles = [TEST_DATA_PATH + label + '/' + wavfile for wavfile in os.listdir(TEST_DATA_PATH + '/' + label)]
    for wavfile in wavfiles:
        mfcc = wav2mfcc(wavfile)
        mfcc_reshaped = mfcc.reshape(1, 20, 11, 1)
        predicted_label = labels[np.argmax(model.predict(mfcc_reshaped))]
        print("Predicted label: ", predicted_label, ". Ground truth: ", label)
        if predicted_label==label:
            counter_correctly_predicted +=1
        counter_total += 1
print("Classified", counter_correctly_predicted,
      "of", counter_total, "=",
      (float(counter_correctly_predicted) / counter_total)*100.0,
      "% of audio files correctly.")

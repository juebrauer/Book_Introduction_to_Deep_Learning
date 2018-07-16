import numpy as np
import librosa
import matplotlib.pyplot as plt
from playsound import playsound
from os import listdir
from os.path import isfile, join
from som import som
from scipy.fftpack import dct


# SOM parameters:
# square root of NR_NEURONS should be an integer!
NR_NEURONS = 7*7
# we want to differ between 10 classes
NR_CLASSES = 10
LEARN_RATE = 0.2
ADAPT_NEIGHBORS = True
TRAIN_STEPS = 5000
RAW_DATA_LEN = 20000
FEATURE_VEC_LEN = 60
SHOW_INFOS_EACH_N_STEP = 1000
CLASSIFY_WINDOW_STEP = 100


def get_feature_vector_from_raw_data(raw_data):
    # compute MFCC features
    # Mel Frequency Cepstral Coefficents (MFCCs)
    # is a feature representation widely used in
    # speech and speaker recognition
    feature_vec = np.mean(librosa.feature.mfcc(y=raw_data, sr=22050, n_mfcc=60).T, axis=0)
    return feature_vec



class audio_dataset:

    def __init__(self, folder):

        self.folder = folder


    def read(self):

        train_folder = self.folder + "/train"
        test_folder  = self.folder + "/test"

        train_files = [f for f in listdir(train_folder)
                       if isfile(join(train_folder, f))]
        test_files  = [f for f in listdir(test_folder)
                       if isfile(join(test_folder, f))]

        train_audio_streams = []
        test_audio_streams  = []

        for f in train_files:
            print("Reading train audio file: ", f)
            data, sampling_rate =\
                librosa.load( train_folder + "/" + f )
            #print("\tsampling rate", sampling_rate)
            #print("\tdata = ", data)
            # plt.plot(data)
            # plt.show()
            #print("\ttype of data is", type(data))
            #print("\tshape of data is", data.shape)
            train_audio_streams.append( data )

        for f in test_files:
            print("Reading test audio file: ", f)
            data, sampling_rate =\
                librosa.load(train_folder + "/" + f)
            test_audio_streams.append(data)

        return train_audio_streams, test_audio_streams



# 1. read in all the training audio files
#    and the test audio files
my_dataset = audio_dataset("10x10_audio_dataset")
train_audio_streams, test_audio_streams = my_dataset.read()

nr_train_audio_streams = len(train_audio_streams)
nr_test_audio_streams  = len(test_audio_streams)
print("I have read in ", nr_train_audio_streams,
      " audio streams for training.")
print("I have read in ", nr_test_audio_streams,
      " audio streams for testing.")


# 2. generate a SOM
my_som = som(FEATURE_VEC_LEN, NR_NEURONS, NR_CLASSES)
my_som.initialize_neuron_weights_to_origin()



# 3. now train a SOM with vectors from the audio streams
for train_step_nr in range(TRAIN_STEPS):

    # choose randomly one of the training audio streams
    audio_nr = np.random.randint(nr_train_audio_streams)

    # choose a random start position in that audio stream
    data = train_audio_streams[audio_nr]
    len_data = len(data)
    start_pos = np.random.randint(len_data - RAW_DATA_LEN)

    # get the data starting at that position
    raw_data = data[start_pos:start_pos+RAW_DATA_LEN]
    train_vec = get_feature_vector_from_raw_data(raw_data)
    #print("train_vec = ", train_vec)
    #print("type of train_vec is ", type(train_vec))

    # train the SOM with this vector
    my_som.train( train_vec, LEARN_RATE, ADAPT_NEIGHBORS, audio_nr)

    # output debug information from time to time ...
    if (train_step_nr % SHOW_INFOS_EACH_N_STEP == 0):
        print("training step:", train_step_nr,
              "choosed audio:", audio_nr,
              "start sample:", start_pos,
              "BMU is:", my_som.BMU_nr)


# 4. now show for all neurons the class counter arrays
for neuron_nr in range(NR_NEURONS):

    print("neuron #", neuron_nr, ":")
    my_som.list_neurons[neuron_nr].show_class_counters()


# 5. now try to classify each test audio stream
for audio_nr in range(nr_test_audio_streams):

    print("\nClassifying test audio stream", audio_nr, ":")

    # get that audio stream
    data = test_audio_streams[audio_nr]
    len_data = len(data)
    print("Len of audio stream is", len_data, "many samples.")

    # classify the audio stream
    votes = np.zeros(NR_CLASSES, dtype=np.int)
    for start_pos in range(0,len_data-RAW_DATA_LEN,CLASSIFY_WINDOW_STEP):

        # get the data starting at that position
        raw_data = data[start_pos:start_pos + RAW_DATA_LEN]
        input_vec = get_feature_vector_from_raw_data(raw_data)

        # feed that input vector into the SOM
        my_som.compute_bmu(input_vec)

        # increase votes
        votes += my_som.list_neurons[my_som.BMU_nr].class_counters
        #class_according_to_bmu = np.argmax(my_som.list_neurons[my_som.BMU_nr].class_counters)
        #votes[class_according_to_bmu] += 1

    # show the votes for the different classes
    for class_nr in range(NR_CLASSES):
        print(votes[class_nr], end=" ")
    print("Seems to be class: ", np.argmax(votes))




#playsound( fname )

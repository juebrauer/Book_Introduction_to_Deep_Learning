# file: dataset_reader.py
#
# Contains helper class dataset
# which is able to return mini batches of a desired
# size of <label, image> pairs
#
# For this the class constructor needs to be passed
# a imagePath in which it expects a subfolder for each
# image category.
#
# It automatically will traverse each subfolder recursively
# in order to generate a image list of the form
# e.g. [['cow', 'cow8439.jpeg'], ['dog', 'dog02.jpeg'], ...]
#
# Images will be read only using OpenCV's imread() function
# when preparing a mini-batch.
# They are not loaded all at once which is good,
# if we want to train on several 10.000 of images.
# (e.g. 1024 images of 1MB each --> 1 GB already!)
#
# The code here is inspired by:
# Wang Xinbo's AlexNet implementation:
# see https://github.com/SidHard/tfAlexNet


import numpy as np
import os
import cv2

RESIZE_WIDTH, RESIZE_HEIGHT = 227,227

class dataset:

    '''
    Checks which image categories are stored
    in the subfolder of <imagePath>:
    e.g. 'cows', 'dogs'
    Then prepares lists of filenames and label information
    for the image files found in the imagePath 
    '''
    def __init__(self, imagePath, extensions):

        # 1. prepare image list with category information
        #    self.data = [['cow', 'cowimg01.jpeg'],
        #                 ['dog', 'dogimg3289.jpeg], ...]
        print("\n")
        print("Searching in folder", imagePath, "for images")
        self.data = createImageList(imagePath, extensions)
        NrImgs = len(self.data)
        print("Found", NrImgs, "images")
        print("Here are the first 5 and the last 5 images and "
              "their corresponding categories I found:")
        for i in range(0,5):
                print(self.data[i])
        for i in range(NrImgs-5,NrImgs):
                print(self.data[i])
        
        # 2. shuffle the data
        np.random.shuffle(self.data)
        self.num_images = len(self.data)
        self.next_image_nr = 0

        # 3. use zip function to unzip the data into two lists
        #    see https://docs.python.org/3.3/library/functions.html#zip
        self.labels, self.filenames = zip(*self.data)

        # 4. show some random images
        for i in range(0,5):
                rnd_idx = np.random.randint(NrImgs)
                rnd_filename = self.filenames[ rnd_idx ]
                print("random filename = ", rnd_filename)
                img = cv2.imread( rnd_filename )   
                img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))
                img_name = "example image " + str(i)
                cv2.imshow(img_name, img)
                cv2.moveWindow(img_name, 300+i*250,100);
        cv2.waitKey(5000)
        cv2.destroyAllWindows()


        # 5. get a list of all categories,
        #    e.g. ['cows', 'dogs']
        category_list = np.unique(self.labels)

        # 6. how many categories are there?
        self.num_labels = len(category_list)
        
        # 7. prepare a dictionary to map
        #   category names to category numbers
        self.category2label = \
            dict(zip(category_list, range(len(category_list))))

        # 8. and the other way round:
        #    prepare a dictionary {} to map category numbers
        #    to category names
        self.label2category =\
            {l: c for c, l in self.category2label.items()}

        # 9. prepare list of ground truth labels
        #    where we can find the ground truth label for
        #    image i at the i-th position in the list
        self.labels = [self.category2label[l] for l in self.labels]


    '''
    Returns the number of images
    available by this dataset object
    '''
    def __len__(self):
        return self.num_images

    '''
    Returns a onehot NumPy array,
    where all entries are set to 0
    but to 1 for the right category
    '''
    def onehot(self, label):
        v = np.zeros(self.num_labels)
        v[label] = 1
        return v


    '''
    Are there further images available?
    '''
    def hasNextRecord(self):
        return self.next_image_nr < self.num_images


    '''
    Resizes the specified OpenCV image to a fixed size    
    Converts it to a NumPy array
    Converts the values from [0,255] to [0,1]
    '''
    def preprocess(self, img):

        # preprocess image by resizing it to 227x227
        pp = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))

        # and convert OpenCV representation to Numpy array
        # note: asarray does not copy data!
        #       see 
        pp = np.asarray(pp, dtype=np.float32)

        # map values from [0,255] to [0,1]
        pp /= 255

        # prepare array of shape width x height x 3 array
        pp = pp.reshape((pp.shape[0], pp.shape[1], 3))
        return pp


    '''
    Returns a (label, image) tuple
     where label is a one-hot teacher vector (list)
     e.g. [0,1] if there are two categories
     and
     image is a NumPy array
     of shape (width, height, 3)
    '''
    def get_next_record(self):

        # will return the next training pair
        # consisting of the input image and a
        # one-hot/teacher label vector
        if not self.hasNextRecord():

            # Ups! We are at the end of the image list!

            # So generate new random order of images

            # randomly shuffle the data again
            np.random.shuffle(self.data)
            self.next_image_nr = 0
            self.labels, self.filenames = zip(*self.data)
            category = np.unique(self.labels)
            self.num_labels = len(category)
            self.category2label = dict(zip(category, range(len(category))))
            self.label2category = {l: c for c,
                                   l in self.category2label.items()}
    
            # prepare ground-truth label information for all images
            # according to the newly shuffled order of the images
            self.labels = [self.category2label[l] for l in self.labels]

        # prepare one-hot teacher vector for the output neurons
        label = self.onehot(self.labels[self.next_image_nr])

        # read in the image using OpenCVs imread()
        # function and then preprocess it
        # (i.e., resize it, convert it to a NumPy array,
        #  convert values from [0,255] to [0,1])
        img_filename = self.filenames[self.next_image_nr]
        img_as_np_array = self.preprocess(cv2.imread(img_filename))

        # prepare next image nr to return
        self.next_image_nr += 1

        # prepare a (label, image) tuple
        return label, img_as_np_array


    '''
    Given a batch size, this function
    first creates a list of (label, image)
    tuples called <records>
    [(0, img-of-cow), (1, img-of-dog), (0, img-of-cow), ...]
    and then returns the labels and images as separate
    tuples
    '''
    def nextBatch(self, batch_size):

        # creates a mini-batch of the desired size
        records = []
        for i in range(batch_size):
            record = self.get_next_record()
            if record is None:
                break
            records.append(record)
        labels, imgs = zip(*records)
        return labels, imgs


'''
Helper function to provide a list of
all label info and image files in all
subfolders in the the given <imagePath>
'''
def createImageList(imagePath, extensions):

    # 1. start with an empty list of labels/filenames
    labels_and_filenames = []

    # 2. each subfolder name in imagePath is considered to be
    #    a class label in stored in categoryList
    categoryList = [None]
    categoryList = [c for c in sorted(os.listdir(imagePath))
                    if c[0] != '.' and
                    os.path.isdir(os.path.join(imagePath, c))]

    # 3. for each of the categories
    for category in categoryList:
        print("subfolder/category found =", category)
        if category:
            walkPath = os.path.join(imagePath, category)
        else:
            walkPath = imagePath
            category = os.path.split(imagePath)[1]

        # create a generator
        w = _walk(walkPath)

        # step through all directories and subdirectories
        while True:

            # get names of dirs and filenames of current dir
            try:
                dirpath, dirnames, filenames = next(w)
            except StopIteration:
                break

            # don't enter directories that begin with '.'
            for d in dirnames[:]:
                if d.startswith('.'):
                    dirnames.remove(d)

            dirnames.sort()

            # ignore files that begin with '.'
            filenames =\
                [f for f in filenames if not f.startswith('.')]
            # only load images with the right extension
            filenames =\
                [f for f in filenames if os.path.splitext(f)[1].lower()
                 in extensions]
            filenames.sort()

            for f in filenames:
                labels_and_filenames.append(
                    [category, os.path.join(dirpath, f)])

    # labels_and_filenames will be a list of
    # two-tuples [category, filename]
    return labels_and_filenames


def _walk(top):
    """
    This is a (recursive) directory tree generator.
    What is a generator?
    See:
    http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    In short:
    - generators are iterables that can be iterated only once
    - their values are not stored in contrast e.g. to a list
    - 'yield' is 'like' return    
    """

    # 1. collect directory names in dirs and
    #    non-directory names (filenames) in nondirs
    names = os.listdir(top)
    dirs, nondirs = [], []
    for name in names:
        if os.path.isdir(os.path.join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)

    # 2. "return" information about directory names and filenames
    yield top, dirs, nondirs

    # 3. recursively process each directory found in current top
    #    directory
    for name in dirs:
        path = os.path.join(top, name)
        for x in _walk(path):
            yield x

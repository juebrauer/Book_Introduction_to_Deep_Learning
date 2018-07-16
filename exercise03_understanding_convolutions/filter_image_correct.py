import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Your Python version is: " + sys.version)
print("Your OpenCV version is: " + cv2.__version__)


x = np.array([[1.0,2.0], [-3.0,-2.0]], dtype='float')


#cap = cv2.VideoCapture('V:/01_job/12_datasets/test_videos/video_testpattern.mp4')
cap = cv2.VideoCapture(0)

kernel = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
'''
kernel = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]])
'''
min_val = -255.0*3.0
max_val = +255.0*3.0




# Gaussfilter:
kernel = np.ones(9*9).reshape(9,9)
min_val = 0.0
max_val = 81.0*255.0






print("Data type of NumPy matrix is ", kernel.dtype)
kernel = np.float32(kernel)
print("Data type of NumPy matrix is now", kernel.dtype)

while(cap.isOpened()):

    # try to get another image from the video
    # capture device
    ret, frame = cap.read()
    if (ret == False):
        break

    # convert image to a grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print("type of gray is ", type(gray))

    # resize grayscale image
    #gray = cv2.resize(gray,None,fx=1.75, fy=1.75, interpolation = cv2.INTER_CUBIC)
    gray = cv2.resize(gray, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_CUBIC)

    # convert from type uint8 to float32
    input_matrix = np.float32(gray)

    # show image information
    print("input_matrix -> shape: ", input_matrix.shape, " depth: ", input_matrix.dtype)

    # show image
    cv2.imshow('frame', gray)

    # filter image with a 2D filter
    output_matrix = cv2.filter2D(input_matrix, -1, kernel)
    print("output_matrix --> shape: ", output_matrix.shape, " Depth: ", output_matrix.dtype)
    print(output_matrix)
    print("Min value = ", output_matrix.min(), " Max value =", output_matrix.max())

    # now convert possible range from [minval, maxval] to [0,1]

    # convert to range [-1,1]
    output_matrix /= max_val

    # convert to range [0,2]
    output_matrix += 1

    # convert to range [0,1]
    output_matrix /= 2.0
    #print("Min value = ", output_matrix.min(), " Max value =", output_matrix.max())

    # show filter result
    cv2.imshow('output_matrix', output_matrix)

    # wait for a key
    c = cv2.waitKey(1)

    # 'q' pressed?
    if (c==113):
        break

cap.release()


'''
IMPORTANT: 
(i) Understand in which range the resulting filter values can live
AND
(ii) Understand how imshow() works

The function imshow displays an image in the specified window. If the window was created with the CV_WINDOW_AUTOSIZE
flag, the image is shown with its original size. Otherwise, the image is scaled to fit the window. The function may
scale the image, depending on its depth:

        If the image is 8-bit unsigned, it is displayed as is.
        If the image is 16-bit unsigned or 32-bit integer, the pixels are divided by 256. That is, the value range
        [0,255*256] is mapped to [0,255].
        If the image is 32-bit floating-point, the pixel values are multiplied by 255. That is, the value range [0,1]
        is mapped to [0,255].
        
        
Also see:

https://stackoverflow.com/questions/25106843/opencv-how-imshow-treat-negative-values

"Now to answer you question:

    What if my Matrix contain negative value in 32-bit floating point. How it will treat it?

The negative values will be displayed as if they are 0."

'''

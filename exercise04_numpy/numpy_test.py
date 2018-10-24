# Exercise 03: NumPy

import numpy as np

#########
# Version
#########

print("Your NumPy version is", np.version.version)


###################
# Generating arrays
###################

# a)
a1 = np.array([2,4,6,8,10])
print("a1=",a1)
print("Numbers in a1 are stored using data type", a1.dtype)

# b)
a1 = a1.astype(np.float32)
print("Numbers in a1 are now stored using data type", a1.dtype)

# c)
a2 = np.array([[1,2],[3,4],[5,6]])
print("a2=",a2)
print("number of dimensions of a2: ", a2.ndim )
print("number of rows    of a2:", a2.shape[0])
print("number of columns of a2:", a2.shape[1])

# d)
print("nr of bytes needed to store one a2 array element :", a2.itemsize)
print("nr of bytes needed to store all a2 array elements:", a2.nbytes)

# e)
a3 = np.array( [[[1,2,3,4],
                 [5,6,7,8],
                 [9,10,11,12]
                ],
                [[13,14,15,16],
                 [17,18,19,20],
                 [21,22,23,24]
                 ]
                ]
              )
print("a3=",a3)
print("number of dimensions of a3: ", a3.ndim )
print("number of slices  of a3:", a3.shape[0])
print("number of rows    of a3:", a3.shape[1])
print("number of columns of a3:", a3.shape[2])


##########################
# Accessing array elements
##########################

# a)
print("Value of that element is ", a3[1,2,3])

# b)
a3[1,2,3]=42
print("Value of that element is now", a3[1,2,3])

# c)
a4 = a3[0,:,:]
print("a4=",a4)

# d)
a5 = a4[:,2]
print("a5=",a5)

# e)
a6 = a4[1,:]
print("a6=",a6)

# f)
a7 = a4[1:3,1:3]
print("a7=",a7)


##################
# Reshaping arrays
##################

# a)
#A = np.array([1,2,3,4,5,6,7,8,9,10])
A = np.array([i for i in range(1,11)])
print("A=",A)
B = A.reshape(2,5)
print("B=",B)

# b)
C = B.reshape(5,2)
print("C=",C)



############################
# Linear algebra with arrays
############################

# a)
A = np.array( [[1,1],
               [0,1]])
B = np.array( [[2,0],
               [3,4]])
print("A=\n",A)
print("B=\n",B)
print("A+B=\n", A+B)
print("elementwise multiplication A*B=\n", A*B)
print("matrix multiplication A*B=\n", np.matmul(A,B))

# b)
A = np.array([[1.0, 2.0],
              [3.0, 4.0]])
print("A=\n",A)
A_inv = np.linalg.inv(A)
print("A_inv=\n", A_inv )
print("A * A_inv=\n", np.matmul(A,A_inv))

# c)
I = np.eye(2,2)
print("I=", I)

# d)
C = np.matmul(A,A_inv)
C = np.round(C)
#C = C.astype(int)
print("C=\n",C)
I = np.eye(2,2)
#I = I.astype(int)
print("I=\n",I)
print("A*A_inv == I? -->", np.array_equal(C,I))



###############
# Random arrays
###############

# a)
rndA = np.random.uniform(low=-1.0, high=1.0, size=(5,3))
print("rndA=\n",rndA)

# b)
rndB = np.random.randint(low=-1, high=1, size=(5,3))
print("rndB=\n",rndB)

# c)
mu = 5.0
sigma = 1.0
rndC = np.random.normal(mu,sigma,1000)
print(rndC)
import matplotlib.pyplot as plt
plt.hist(rndC, 50)
plt.show()





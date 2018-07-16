# PYTHON 3 CODE
import sys

print("Your Python version is: " + sys.version)



# Difference #1: integer division
# Division operator always yields a float as result
# Python 2 treats numbers that you type without any digits after the decimal point as integers, Python 3 not
print('3 / 2   =', 3 / 2)
print('3 / 2.0 =', 3 / 2.0)
print('3 / 1   =', 3 / 1)


# Difference #2: raising exceptions
#raise IOError, "This is an error message"  #INVALID SYNTAX in Python3
#raise IOError("This is an error message")  #Syntax ok in Python3


# Difference #3: handling execeptions
try:
    a = b + c
except NameError as err:
    print(err, '--> Here is another error message')


# Difference #4: List comprehension loop variables live NOT in a global namespace
i=42
print("i="+str(i))
list1 = [0,1,2,3,4]
list2 = [i*i for i in list1]
print(list2)
print("i="+str(i))



# Difference #5: use raw_input() to read in a string
name = input( "What is your name? " )
print ("Hello " + name)




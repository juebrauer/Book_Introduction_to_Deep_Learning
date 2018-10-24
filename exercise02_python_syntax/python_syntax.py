# Exercise 02: Getting used to Python's syntax

##################
print("\n\nLoops")
##################

print("\na)")
#############
for i in range(10,30,3):
    print(i, end=' ')
print()

i=10
while i<30:
    print(i, end=' ')
    i+=3
print()



print("b)")
#############
counter=0
while(counter<10):
    print(counter, end=' ')
    counter +=1
else:
    print("counter=" + str(counter))

for i in range(1, 10):
    if(i%5==0):
        break
    print(i, end=' ')
else:
    print("i=" + str(i))



###########################
print("\n\nDynamic typing")
###########################

print("a)")
a = 2
b = 3.1
c = 'd'
d = "a string"
e = [1,22,333]
f = (4,55,666)
print("data type of a is : ", type(a))
print("data type of b is : ", type(b))
print("data type of c is : ", type(c))
print("data type of d is : ", type(d))
print("data type of e is : ", type(e))
print("data type of f is : ", type(f))

e.append(4444)
e.append([55555])
print("Updated list e is now : " + str(e))


print("b)")
b = 21
b = b+b
print(b)
b = "3"
b = b+b
print(b)
print(type(int(b)))
print(type(str(type(int(b)))))
print(str(type(str(type(int(b)))))[3])



#######################
print("\n\nSelections")
#######################

# while True:
#  inp = input()
#  try:
#      inp = float(inp)
#  except ValueError:
#      if inp=="exit":
#          break
#      else:
#          print("Invalid command!")
#  else:
#      if inp>=1 and inp<=3:
#          print("A")
#      elif inp>=4 and inp<=6:
#          print("B")
#      elif inp>=7 and inp<=9:
#          print("C")
#      else:
#          print("Invalid number!")



######################
print("\n\nFunctions")
######################

print("a)")
def f1(value1,value2):
    return value1+value2, value1*value2

result1 = f1(10,5)
print(result1)
print(type(result1))


print("b)")
def f2(value1=2,value2=3):
    return value1+value2, value1*value2

print(f2())
print(f2(value2=4))


##################
print("\n\nLists")
##################

print("a)")
shoppinglist = ['cheese', 'milk', 'water']
shoppinglist.append('apples')
print(shoppinglist)
shoppinglist.remove('milk')
print(shoppinglist)
for tobuy in shoppinglist:
    print("I have to buy " + tobuy)

print("b)")
from math import pi
list_pi_rounded = [(round(pi, i)) for i in range(1, 6)]
print(list_pi_rounded)



####################
print("\n\nClasses")
####################

print("a)")
class car:

    speed = 0.0
    mileage = 0.0
    maxspeed = 0.0
    carname = ""

    def __init__(self, carname, maxspeed):
        self.carname  = carname
        self.maxspeed = maxspeed

    def set_speed(self, newspeed):
        self.speed = newspeed
        if newspeed > self.maxspeed:
            print("Speed "+ str(newspeed) + "? "
                  "Come on! This is a '" + self.carname + "'!")

    def drive(self, hours):
        self.mileage = self.mileage + hours*self.speed

    def show_status(self):
        print("car '" + self.carname + "'"
              " currently drives with speed " + str(self.speed) +
              " and has mileage " + str(self.mileage))


c1 = car("old car", 90)
c2 = car("sports car", 300)

c1.set_speed(80)
c1.drive(2)
c1.show_status()

c2.set_speed(200)
c2.drive(2)
c2.show_status()

c1.set_speed(200)


print("b)")
class convertible(car):

    def __init__(self, name, maxspeed, time_to_open_roof):
        car.__init__(self, name, maxspeed)
        self.time_to_open_roof = time_to_open_roof

    def show_status(self):
        car.show_status(self)
        print("Time to open roof: " + str(self.time_to_open_roof))

c3 = convertible("your convertible", 250, 5)
c3.show_status()


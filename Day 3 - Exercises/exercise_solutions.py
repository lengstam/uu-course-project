""" Exercise 1 """

class Person():
    def __init__(self, first, last):
        self.firstname = first
        self.lastname = last
    def __str__(self): #call using print()
        return "%s %s" % (self.firstname, self.lastname)
    
class Student():
    def __init__(self, first, last, subject):
        Person.__init__(self, first, last)
        self.subject = subject
    def printNameSubject(self):
         return "%s %s, %s" % (self.firstname, self.lastname, self.subject)
        
class Teacher(): #tried having a fix name of the teacher
    def __init__(self, course):
        self.course = course
        if course == 'Python programming':
            Person.__init__(self, 'Filipe', 'Maia')
        else:
            print('This course has no teacher')
    def printNameCourse(self):
        return"%s %s, %s" % (self.firstname, self.lastname, self.course)
        
""" Exercise 2 """

import numpy as np

a = np.zeros(10)
a[4] = 1

b = np.arange(10,50)

c = np.flip(b)

d = np.arange(9).reshape(3,3)

e = np.array([1,2,0,0,4,0])
e1 = np.nonzero(e)

f = np.random.random(30)
f_mean = np.mean(f)

g = np.ones((4,4))
g[1:-1,1:-1] = 0

h = np.zeros((8,8))
h[::2,::2] = 1 # Ones starting on first row, first column
h[1::2,1::2] = 1 # Ones starting on second row, second column

i = np.array([[1,0],[0,1]])
i1 = np.tile(i,(4,4))

j = np.arange(11)
j[3:9] *= -1
print(j)

k = np.random.random(10)
k.sort()
print(k)

l1 = np.random.randint(0,2,5)
l2 = np.random.randint(0,2,5)
equal = l1 == l2
print(equal)

m = np.arange(10, dtype=np.int32)
print(m.dtype)
m = m.astype(np.float32)
print(m.dtype)

n1 = np.arange(9).reshape(3,3)
n2 = n1 + 1
n3 = np.dot(n1,n2)
n4 = np.diag(n3)
print(n4)


""" Exercise 3 """

# See matmult_improved.py

# In the last exercise I improved it by doing the matrix multiplication using numpy.
# Now, I also used numpy's randomizer to produce both X and Y vectors, and then simply print the whole array.
# This reduced total time to 0.02 seconds, compared to 16.25 seconds using the original code (99.88 % reduction)


""" Exercise 4 """

# See the mpi_ranks.py and mpi_sum.py files for my proposed solutions.


""" Exercise 5 """

# Implemented the following code:

# import numpy as np

# def ftrans(n):
#   a = np.random.randn(n,n).astype(np.float32)
#   b = np.fft.fft2(a)

# %timeit ftrans(32)

# In my case, an array size of 32x32 was needed for regular numpy to be faster.
# The outcome was the same using float32, although the overall time consumption was lower.

# Results float64: (numpy time/cupy time [ms])
    # 32: 0.069/ 0.140
    # 64: 0.203/0.142 (0.231)
    # 128: 0.687/0.109
    # 256: 2.85/0.111
    # 512: 20.5/0.36
    # 1024: 71.5/1.52
    # 2048: 349/5.96

# Results float32: (numpy time/cupy time [ms])
    # 32: 0.064/ 0.135
    # 64: 0.195/0.126
    # 128: 0.682/0.129
    # 256: 3.37/0.131
    # 512: 13.4/0.151
    # 1024: 62.7/0.728
    # 2048: 287/0.181
    





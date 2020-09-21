import sys
import numpy as np

a = np.array([1, 2, 3])
print('1D array: ')
print(a)
print()
b = np.array([(1, 2, 3), (4, 5, 6)])
print('2D array: ')
print(b)

s = range(1000)
print('El resultado en lista es: ')
print(sys.getsizeof(5)*len(s))
print()
d = np.arange(1000)
print("El resultado NumPy es: ")
print(d.size*d.itemsize)

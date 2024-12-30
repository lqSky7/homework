import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

a_slice = a[1:4]
b_slice = b[0:2, 1:3]
c_slice = c[1, 0, :]

print(a)
print(b)
print(c)
print(a_slice)
print(b_slice)
print(c_slice)


import pandas as pd

a = pd.Series([1, 2, 3, 4, 5])
b = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
c = pd.Series({'x': 100, 'y': 200, 'z': 300})

a_element = a[2]
b_element = b['b']
c_element = c['y']

print(a)
print(b)
print(c)
print(a_element)
print(b_element)
print(c_element)

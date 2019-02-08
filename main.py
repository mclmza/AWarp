import numpy as np
import awarp

s = np.array([1, 2, 3, -1, 1])
t = np.array([1, -2, 4, 1])

print(awarp.awarp(s, t))
print(awarp.awarp(s, t, w=4))
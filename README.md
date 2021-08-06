# Awarp
Python implementation of [AWarp](https://www.cs.unm.edu/~mueen/Projects/AWarp/awarp.pdf) algorithm using numba to optimize machine code at runtime.
Awarp is a dynamic time warp derivate optimized for sparse data. 

## Usage
Given 2 run length encoded time series:

Runs of zeroes are encoded as negative values:

````python
 s_full = np.array([1,2,3,0,1])
 t_full = np.array([1,0,0,4,1])

 # are equivalent to 
 s = np.array([1, 2, 3, -1, 1])
 t = np.array([1, -2, 4, 1])
````

To calculate global AWarp distance:

`awarp.awarp(s, t)`

To calculate constrained AWarp distance:

`awarp.awarp(s, t, w=4)`

where `w` is the size of the window in number of points.
For constrained calculation the dataset has to being with an event (a positive non zero value).

#### License

MIT
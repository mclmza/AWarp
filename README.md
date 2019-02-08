# Awarp
Python implementation of [AWarp](https://www.cs.unm.edu/~mueen/Projects/AWarp/awarp.pdf) algorithm using numba to optimize machine code at runtime.

## Usage
Given 2 [run length encoded](https://en.wikipedia.org/wiki/Run-length_encoding) time series:

`s = np.array([1, 2, 3, -1, 1])`

`t = np.array([1, -2, 4, 1])`

To calculate global AWarp distance:

`awarp.awarp(s, t)`

To calculate costrained AWarp distance:

`awarp.awarp(s, t, w=4)`

where `w` is the size of the window in number of points.

#### License

MIT
# PyCuda-KMeans
A parallelized PyCuda implementation of the KMeans clustering algorithm.

## Setup
First, install the CUDA Toolkit on your computer. Downloads and instructions can be found at [https://developer.nvidia.com/cuda-downloads].

Then, install the pycuda library with
  
  pip install pycuda
  
You will also need numpy, if you don't have it, you can install it with
  
  pip install numpy
  
Verify that the pycuda library is working properly using
  
  python PyCudaCheck.py
  
This piece of code generates the array {1 2 3 ... 398 399 400}, which is sent to the gpu. The array is retrieved from the gpu and printed. Then a function is applied to the array that doubles all the values, and the array is again retrieved from the gpu and printed, the result should be {2 4 6 ... 796 798 800}.

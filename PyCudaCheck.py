import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
__global__ void doublearray(float *a){
    unsigned int i = threadIdx.x;
    a[i] *= 2;
}
""")

array = gpuarray.arange(400, dtype = np.float32)

print(array.get())

doublearray = mod.get_function("doublearray")

doublearray(array, block = (len(array),1,1), grid = (1,1))

print(array.get())
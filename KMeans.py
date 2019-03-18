import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
__global__ void kmeans(float *x, float *y, float *c_x, float *c_y, float **dist, int *cluster, int *cluster_count){
    unsigned int i = blockIdx.x;
    unsigned int j = threadIdx.x;
    dist[i][j] = sqrt(pow(x[i] - c_x[j],2) + pow(y[i] - c_y[j]));
    
    __syncthreads();
    
    if (j == 0){
        float min_centroid = dist[i][0];
        float min_index = 0;
        for (int m = 0; m < blockDim.x; m++){
            if (dist[i][m] < min_centroid){
                min_centroid = dist[i][m];
                min_index = m;
            }
        }
        cluster[i] = min_index;
    }
    
}
__global__ void recompute_clusters(float *x, float *y, float *c_x, float *c_y, float *dist, int *cluster, int *cluster_count){
    unsigned int i = blockIdx.x;
    unsigned int j = threadIdx.x;
    
    if(cluster[j] == i){
        c_x[i] += x[j];
        c_y[i] += y[j];
        cluster_count[i]++;
    }
    
    __syncthreads();
    
    if(j == 0){
        c_x[i] /= cluster_count[i];
    }
}
__global__ void reset_values(float *c_x, float *c_y, int *cluster_count){
    unsigned int i = threadIdx.x;
    
    c_x[i] = 0;
    c_y[i] = 0;
    cluster_count[i] = 0;
}
__global__ void copy_to(int *a, int *b){
    unsigned int i = threadIdx.x;
    b[i] = a[i];
}
""")

kmeans = mod.get_function("kmeans")
recompute = mod.get_function("recompute_clusters")
reset = mod.get_function("reset_values")
copy = mod.get_function("copy_to")

def kmeans_iter(x, y, c_x, c_y, dist, cluster, cluster_old, cluster_count):
    copy(cluster, cluster_old, block = (K,1,1), grid = (1,1))
    kmeans(x, y, c_x, c_y, dist, cluster, cluster_count, block = (K,1,1), grid = (N,1))
    reset(c_x, c_y, cluster_count, block = (K,1,1), grid = (1,1))
    recompute(x, y, c_x, c_y, dist, cluster, cluster_count, block = (N,1,1), grid = (K,1))
def kmeans(x,y,K=3):
    assert len(x) == len(y)
    N = len(x)
    c_x = gpuarray.to_gpu(np.random.random(K).astype(np.float32) * 10)
    c_y = gpuarray.to_gpu(np.random.random(K).astype(np.float32) * 10)
    dist = gpuarray.to_gpu(np.empty((N,K)).astype(np.float32))
    cluster = gpuarray.to_gpu(np.empty(N).astype(np.int32))
    cluster_old = gpuarray.to_gpu(np.empty(N).astype(np.int32))
    cluster_count = gpuarray.to_gpu(np.zeros(K).astype(np.int32))
    while True:
        kmeans_iter(x,y,c_x,c_y,dist,cluster,cluster_old,cluster_count)
        if cluster_old.get() == cluster.get():
            break
    return cluster.get(), (c_x.get(), c_y.get())

if __name__ == "__main__":
    N = 10
    K = 3
    x = np.random.random(N)
    y = np.random.random(N)
    print(kmeans(x,y,K))

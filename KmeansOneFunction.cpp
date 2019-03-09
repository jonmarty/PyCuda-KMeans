__global__ void kmeans(float *x, float *y, float *c_x, float *c_y, float *dist, int *cluster, int *cluster_count){
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

    if(j == 0 && i == 0){
       for(int m = 0; m < c_x.)
    }
}
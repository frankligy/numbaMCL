#!/usr/bin/env python
# coding: utf-8



# Let's load the package
import numpy as np
from numba import cuda




# let's prepare the data
np.random.seed(42)
start = np.random.randint(1,10,(1024,1024))
symmetry = np.tril(start,k=0) + np.tril(start,k=-1).T  # [1024,1024]




# let's design the necessary sequential version functions of MCL algorithm
def add_self_loop(adjacency,loop_value):
    for i in range(adjacency.shape[0]):
        adjacency[i,i] += loop_value
    return adjacency

def l1_normalization_like(adjacency):
    result = np.empty(adjacency.shape)
    for j in range(adjacency.shape[1]):
        result[:,j] = adjacency[:,j] / np.sum(adjacency[:,j])
    return result

def expand(adjacency,expansion):
    result = adjacency ** expansion
    return result

def inflate(adjacency,inflation):
    result = l1_normalization_like(adjacency**2)
    return result

def iterate(adjacency,expansion=2,inflation=2):
    result = inflate(expand(adjacency,expansion),inflation)
    return result

def allclose(matrix1,matrix2,atol=1e-08,rtol=1e-05):
    c = np.abs(matrix1-matrix2) - (atol + rtol*np.abs(matrix2))
    if np.max(c) <= 0:
        return (True,c)
    else:
        return (False,c)




# Now let's dive into the cuda version, the parallel version of MCL
# implement outline:
# 1. write three kernel function: expand, inflate, check_converge
# 2. run the MCL from the host




# Kernel function1 : expand
@cuda.jit
def gpu_expand(in_matrix,out_matrix):  # 2-d block,threads
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    blocksize_x = cuda.blockDim.x
    blocksize_y = cuda.blockDim.y
    pos_x = tx + blocksize_x * bx
    pos_y = ty + blocksize_y * by
    if pos_x <= in_matrix.shape[0] and pos_y <= in_matrix.shape[1]:
        out_matrix[pos_x,pos_y] = in_matrix[pos_x,pos_y] ** 2
        




# kernel function2: inflation
@cuda.jit
def gpu_inflate(in_matrix,out_matrix,final_matrix):  # 1-d block, threads
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    blocksize_x = cuda.blockDim.x
    pos_x =  tx + blocksize_x * bx
    if pos_x <= in_matrix.shape[1]:
        for i in range(out_matrix.shape[0]):
            out_matrix[i,pos_x] = in_matrix[i,pos_x] ** 2
        sum_ = 0
        for i in range(out_matrix.shape[0]):
            sum_ += out_matrix[:,pos_x][i]
        for i in range(out_matrix.shape[0]):
            final_matrix[i,pos_x] = out_matrix[i,pos_x] / sum_  # within-column normalization
        




# kernel function3: check_converge
@cuda.jit
def gpu_check_converge(old_matrix,new_matrix,result_matrix):
    atol = 1e-5
    rtol = 1e-8
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    blocksize_x = cuda.blockDim.x
    blocksize_y = cuda.blockDim.y
    pos_x = tx + blocksize_x * bx
    pos_y = ty + blocksize_y * by
    if pos_x <= old_matrix.shape[0] and pos_y <= old_matrix.shape[1]:
        c = abs(new_matrix[pos_x,pos_y] - old_matrix[pos_x,pos_y]) - (atol + rtol*abs(old_matrix[pos_x,pos_y]))
        if c <= 0: 
            result_matrix[pos_x,pos_y] = True
        else:
            result_matrix[pos_x,pos_y] = False
    
        






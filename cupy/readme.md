# Introduction in CuPy

## Overview
This tutorial demonstrates the simple use of cupy and provides inspiration for benchmarking methods. A CPU/GPU agnostic code example for numpy and cupy is also provided.

## Prerequisites
Before you begin, ensure you have the following installed:
- NVIDIA GPU
- NVIDIA container toolkit
- Docker or podman
- ~7.22 GB container size (CUDA is within container)

## Usage
### Build the docker container
To build the Docker container, run the following command in your terminal within the directory containing the Dockerfile:
```bash
docker build -t cupy:latest .
```
### Run the container without rebuilding
To avoid rebuilding the container every time, you can mount your local directory:
```bash
docker run --gpus all -v ~/Projects/tutorials/IntroPythonParProc/cupy:/app cupy:latest
```
To run the container without GPU support to experience the agnostic code implementation remove `--gpus all`:
```bash
docker run -v ~/Projects/tutorials/IntroPythonParProc/cupy:/app cupy:latest
```
**Use comments to execute the various functions in main().**
## Output
### Simple benchmarking
```bash
my_func             :    CPU:   131.447 us   +/-  9.827 (min:   120.396 / max:   159.227) us     GPU-0:  1523.733 us   +/-  5.237 (min:  1516.160 / max:  1537.632) us
```
The benchmark results show that the CPU execution time for my_func is significantly faster, averaging about 131.447 microseconds, compared to the GPU average of 1523.733 microseconds. This may be due to the overhead of transferring data to and from the GPU and the relatively small size of the input array, making the CPU more efficient for this particular task.

### Medium benchmarking
This benchmark varies based on the time required for data transfer and computation as matrix sizes increase during matrix multiplication. Context initialization has a significant impact on the first calculation in a process. You can see the results in plot.png inside the container.

### CPU/GPU agnostic code sample for numpy and cupy
With GPU support, the computation is done on the GPU with CuPy.

```bash
GPU is available.
Using: cupy
[1.31326169 2.12692801 3.04858735]
```

Without GPU support, the computation is done on the CPU using NumPy.

```bash
An error occurred: cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version
No GPU found.
An error occurred: cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version
Using: numpy
[1.31326169 2.12692801 3.04858735]
```
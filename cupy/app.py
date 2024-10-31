import subprocess
import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt
from cupyx.profiler import benchmark

# Helper function to print installed packages
def print_installed_packages():
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE)
    print(result.stdout.decode())

#------------------------------------------------------
# simple benchmarking
#------------------------------------------------------
def my_func(a):
    return cp.sqrt(cp.sum(a**2, axis=-1))

def benchmarking():
    a = cp.random.random((256, 1024))
    print(benchmark(my_func, (a,), n_repeat=20)) 

#------------------------------------------------------
# medium benchmarking
#------------------------------------------------------

# Context Initialization
def warm_up():
    A = cp.ones((10, 10))
    B = cp.ones((10, 10))
    _ = cp.dot(A, B)

def pipeline():
    Ns, np_times, to_gpu_times, cp_times, to_cpu_times  = [], [], [], [], []
    # have a try with larger ranges
    for N in range(100, 1100, 100):
        A_np = np.random.rand(N, N)
        B_np = np.random.rand(N, N)

        # NumPy matrix multiplication
        start_time = time.time()
        C_np = np.dot(A_np, B_np)
        end_time = time.time()
        np_time = end_time - start_time

        # Transfer to GPU
        start_time = time.time()
        A_cp = cp.asarray(A_np)
        B_cp = cp.asarray(B_np)
        end_time = time.time()
        to_gpu_time = end_time - start_time

        # CuPy matrix multiplication
        start_time = time.time()
        C_cp = cp.dot(A_cp, B_cp)
        end_time = time.time()
        cp_time = end_time - start_time

        # Transfer back to CPU      
        start_time = time.time()
        C_np = cp.asnumpy(C_cp)
        end_time = time.time()
        to_cpu_time = end_time - start_time

        # Append times and sizes
        Ns.append(N)
        np_times.append(np_time)
        to_gpu_times.append(to_gpu_time)
        cp_times.append(cp_time)
        to_cpu_times.append(to_cpu_time)
    
    # Calculate total CuPy time
    cp_total_times = np.array(to_gpu_times) + np.array(cp_times) + np.array(to_cpu_times)
    
    # Plotting
    plt.plot(Ns, np_times, label='NumPy time')
    plt.plot(Ns, to_gpu_times, label='to GPU time')
    plt.plot(Ns, cp_times, label='CuPy time')
    plt.plot(Ns, to_cpu_times, label='to CPU time')
    plt.plot(Ns, cp_total_times, label='Total CuPy time')
    plt.legend()
    plt.xlabel('Matrix size N')
    plt.ylabel('Time in s')
    plt.savefig('plot.png')

def cpu_vs_gpu():
    # warm_up() # have a try with warmup
    pipeline()

#------------------------------------------------------
# CPU/GPU agnostic code sample for numpy and cupy
#------------------------------------------------------
def is_gpu_available():
    try:
        # Check if CUDA is available
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Stable implementation of log(1 + exp(x))
def softplus(x):
    xp = cp.get_array_module(x)  # 'xp' is a standard usage in the community
    print("Using:", xp.__name__)
    return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))

def agnostic_main():
    if is_gpu_available():
        print("GPU is available.")
    else:
        print("No GPU found.")

    x = [1,2,3]
    
    a = np.array(x)
    if is_gpu_available():
        a = cp.asarray(a)
    print(softplus(a))

    # as one-liner
    # b = cp.asarray(np.array(x)) if is_gpu_available() else np.array(x)
    # print(softplus(b))

#------------------------------------------------------
# main() function
#------------------------------------------------------

def main():
    print('start main()')
    print_installed_packages()
    benchmarking()
    # cpu_vs_gpu()
    # agnostic_main()
    print('main() finished')

if __name__ == '__main__':
    main()
# Parallel model training on CPU in Python

## Overview
This tutorial demonstrates how to efficiently train a Random Forest model using parallelization techniques in Python. We will explore two popular libraries: `joblib` and `multiprocessing`. The goal is to showcase how to leverage multiple CPU cores to speed up model training, especially for computationally intensive tasks.

## Prerequisites
Before you begin, ensure you have the following installed:
- Docker or podman
- ~470 MB container size 

## Dataset
For this tutorial, we will use the **Iris dataset**, a well-known dataset in machine learning. Additionally, we will demonstrate the use of synthetic data generated with `make_classification` for larger datasets.

## Key Components
- **Timer decorator**: A decorator to measure the execution time of functions.
- **Model training function**: A function that trains a Random Forest model and evaluates its accuracy.
- **Sequential training**: A method to train the model sequentially, using a single CPU core.
- **Parallel training with joblib**: A method to train the model in parallel using the `joblib` library.
- **Parallel training with multiprocessing**: A method to train the model in parallel using the `multiprocessing` library.

## Usage
### Build the docker container:
To build the Docker container, run the following command in your terminal within the directory containing the Dockerfile:
```bash
docker build -t mlcpu:latest .
```

### Run the docker container
To run the container, use:
```bash
docker run mlcpu
```
### Run the container without rebuilding
To avoid rebuilding the container every time, you can mount your local directory:
```bash
docker run -v ~/Projects/tutorials/IntroPythonParProc/mlcpu:/app mlcpu
```
## Output
The output will display the training accuracy and time taken for each model configuration (number of estimators) for both sequential and parallel training methods. You will also see the CPU core being utilized during training.

### Example Output
```bash
Starting sequential training...

Training on CPU core: 2 | 10 estimators | Accuracy: 0.9667 | Time: 0.0536 seconds
Training on CPU core: 2 | 50 estimators | Accuracy: 1.0000 | Time: 0.0737 seconds
Training on CPU core: 2 | 100 estimators | Accuracy: 0.8667 | Time: 0.1904 seconds
Training on CPU core: 2 | 200 estimators | Accuracy: 1.0000 | Time: 0.2861 seconds

----------------------------------------

Elapsed time: 0.6051 seconds for sequential_training

Starting parallel training with joblib...

Training on CPU core: 0 | 10 estimators | Accuracy: 0.9667 | Time: 0.0584 seconds
Training on CPU core: 3 | 50 estimators | Accuracy: 1.0000 | Time: 0.1153 seconds
Training on CPU core: 1 | 100 estimators | Accuracy: 1.0000 | Time: 0.1980 seconds
Training on CPU core: 2 | 200 estimators | Accuracy: 0.9667 | Time: 0.3490 seconds

----------------------------------------

Elapsed time: 3.3668 seconds for parallel_training_joblib

Starting parallel training with multiprocessing...

Training on CPU core: 4 | 10 estimators | Accuracy: 0.9333 | Time: 0.0321 seconds
Training on CPU core: 3 | 50 estimators | Accuracy: 0.9667 | Time: 0.1389 seconds
Training on CPU core: 1 | 100 estimators | Accuracy: 0.9667 | Time: 0.2111 seconds
Training on CPU core: 7 | 200 estimators | Accuracy: 0.9667 | Time: 0.3755 seconds

----------------------------------------

Elapsed time: 0.4364 seconds for parallel_training_multiprocessing
```
### Interpretation
The output from the training process provides valuable insights into the performance of the Random Forest model under different configurations and training methods. Here’s a breakdown of the key points:
#### Sequential training
The sequential training results show the accuracy and time taken for various numbers of estimators (10, 50, 100, and 200).
The accuracy generally improves with an increasing number of estimators, although there is a slight drop at 100 estimators, which may indicate overfitting or variability in the dataset. The total elapsed time for sequential training is 0.6051 seconds.
#### Parallel training with joblib
Different CPU cores are utilized for training various configurations, demonstrating effective workload distribution. However, the elapsed time for parallel training with joblib (3.3668 seconds) is longer than sequential training due to overhead from managing multiple processes.
#### Parallel training with multiprocessing
Similar to `joblib`, this method also shows the utilization of different CPU cores. The elapsed time for parallel training with `multiprocessing` (0.4364 seconds) is significantly lower than that of `joblib`, indicating that this method may have less overhead for the given workload. The accuracy results are consistent with those from the sequential and joblib methods, reinforcing the reliability of the model across different training approaches.


### Important notes
- **Random state**: To ensure variability in model training, consider setting a random state in the Random Forest model.
- **Dataset size**: The Iris dataset is small; for more significant variability in results, consider using larger synthetic datasets. You can use the commented function `make_classification()` in `train_model()` to create a larger dataset based on your requirements.
- **Performance**: The performance of parallelization may vary based on the workload and the overhead of managing multiple processes.
- **CPU core usage**: Depending on the underlying workload manager of the operating system, it is possible for multiple CPU cores to be utilized during sequential training. Modern operating systems often employ CPU scheduling and thread management techniques that can lead to some level of parallel execution, even when a process is designed to run on a single core. As a result, while sequential training primarily uses one core, the OS may allocate resources from other cores to optimize performance, especially for larger workloads. However, this utilization is typically less efficient compared to dedicated parallel training methods.

## Conclusion
This tutorial provides a foundation for understanding how to implement parallel model training in Python. By utilizing `joblib` and `multiprocessing`, you can significantly reduce training time for computationally intensive tasks. Experimenting with different configurations and datasets will help you better understand the performance characteristics of your models and the benefits of parallelization.
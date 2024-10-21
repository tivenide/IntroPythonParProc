import subprocess
import time
import psutil
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from multiprocessing import Pool

# Helper function to print installed packages
def print_installed_packages():
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE)
    print(result.stdout.decode())

# Timer decorator to measure the execution time of functions
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time() # Start the timer
        result = func(*args, **kwargs) # Call the original function
        end_time = time.time() - start_time # Calculate elapsed time
        print(f"Elapsed time: {end_time:0.4f} seconds for {func.__name__}\n") # Print elapsed time
        return result, end_time # Return the result and elapsed time
    return wrapper

# Function to train a Random Forest model
def train_model(n_estimators):
    # Get the current CPU core being used
    current_cpu = psutil.Process().cpu_num()
    start_time = time.time()  # Start timing for this model

    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Create a larger synthetic dataset
    # X, y = make_classification(n_samples=10_000, n_features=20, n_informative=10, n_redundant=10, random_state=42)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize and train the Random Forest model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    # Evaluate the model's accuracy on the test set
    accuracy = model.score(X_test, y_test)

    # Calculate and print the time taken to train the model
    end_time = time.time() - start_time  # End timing for this model
    print(f"Training on CPU core: {current_cpu} | {n_estimators} estimators | "
          f"Accuracy: {accuracy:.4f} | Time: {end_time:0.4f} seconds")

    return accuracy # Return the accuracy of the model

# Sequential implementation of model training
@timer
def sequential_training(n_estimators_list):
    results = [] # List to store results
    print("Starting sequential training...\n")
    for n in n_estimators_list:
        accuracy = train_model(n) # Train the model with the current number of estimators
        results.append((n, accuracy)) # Store the results
    print("\n" + "-" * 40 + "\n")
    return results

# Parallel implementation using joblib
@timer
def parallel_training_joblib(n_estimators_list):
    print("Starting parallel training with joblib...\n")
    results = Parallel(n_jobs=-1)(delayed(train_model)(n) for n in n_estimators_list)
    print("\n" + "-" * 40 + "\n")
    return results

# Parallel implementation using multiprocessing
@timer
def parallel_training_multiprocessing(n_estimators_list):
    print("Starting parallel training with multiprocessing...\n")
    # Create a pool of worker processes
    with Pool() as pool:
        results = pool.map(train_model, n_estimators_list)  # Map the train_model function to the list
    print("\n" + "-" * 40 + "\n")
    return results  # Return the results


if __name__ == '__main__':
    # List of different numbers of estimators to test
    n_estimators_list = [10, 50, 100, 200]  # Increased complexity

    # Run sequential training and store results
    sequential_results = sequential_training(n_estimators_list)

    # Run parallel training and store results
    parallel_results = parallel_training_joblib(n_estimators_list)

    # Run parallel training with multiprocessing and store results
    multiprocessing_results = parallel_training_multiprocessing(n_estimators_list)

import sys
import numpy as np
# from mykmeanssp import fit
DEFAULT_ITER = 300
DEFAULT_ERROR_MESSAGE = "An Error Has Occurred"
CLUSTERS_ERROR_MESSAGE = "Invalid number of clusters!"
ITER_ERROR_MESSAGE = "Invalid maximum iteration!"
EPS_ERROR_MESSAGE = "Invalid epsilon!"

def exit_with_error(message):
    print(message)
    sys.exit(1)

def validate_args(args):
    # Validate argument count
    if len(args) not in [5,6]:
        exit_with_error(DEFAULT_ERROR_MESSAGE)

    # Validate k is a natural number
    try:
        k_float = float(args[1])
        if not k_float.is_integer():
            raise ValueError
        k = int(k_float)
        if k <= 1:
            raise ValueError
    except ValueError:
        exit_with_error(CLUSTERS_ERROR_MESSAGE)

    # Validate iter is a natural number
    if len(args) == 2:
        try:
            iter_float = float(args[1])
            if not iter_float.is_integer():
                raise ValueError
            maximum_iteration = int(iter_float)
        except ValueError:
             exit_with_error(ITER_ERROR_MESSAGE)
        eps_idx = 3

    else:
        maximum_iteration = DEFAULT_ITER
        eps_idx = 2

     # Validate 1 < iter < 1000
    if not (1 < maximum_iteration < 1000):
        exit_with_error(ITER_ERROR_MESSAGE)

    # Validate epsilon
    try:
        if not args[eps_idx].replace(".", "", 1).isdigit():
            raise ValueError
        eps = float(args[eps_idx])
        if not (eps >= 0):
            raise ValueError
    except ValueError:
        exit_with_error(EPS_ERROR_MESSAGE)

    return k, maximum_iteration, eps, args[eps_idx + 1], args[eps_idx + 2]

def read_and_merge_files(file_name_1, file_name_2):
    # Load data from files
    try:
        data_points_1 = np.loadtxt(file_name_1, delimiter=",", dtype=np.float64)
        data_points_2 = np.loadtxt(file_name_2, delimiter=",", dtype=np.float64)

        # Perform inner join by first column
        indices = np.intersect1d(data_points_1[:, 0], data_points_2[:, 0])
        data_points_1 = data_points_1[np.isin(data_points_1[:, 0], indices)]
        data_points_2 = data_points_2[np.isin(data_points_2[:, 0], indices)]

        # Sort each array by first column
        data_points_1 = data_points_1[np.argsort(data_points_1[:, 0])]
        data_points_2 = data_points_2[np.argsort(data_points_2[:, 0])]

        # Combine data points
        combined_data_points = np.hstack((data_points_1, data_points_2[: , 1:]))

        # Drop first column
        combined_data_points = combined_data_points[:, 1:]

        return combined_data_points
    
    except:
        exit_with_error(DEFAULT_ERROR_MESSAGE)
    
def init_centroids(k, data_points):
    np.random.seed(1234)
    n, dim = data_points.shape
    chosen_centroids = []
    chosen_centroids_idx = []

    # Step 1
    chosen_centroids_idx.append(np.random.choice(n))
    chosen_centroids.append(data_points[chosen_centroids_idx[0]])

    # Steps 2-4
    for _ in range(1, k):
        distances = np.min(np.linalg.norm(data_points[:, np.newaxis] - np.array(chosen_centroids), axis=2), axis=1)
        probabilities = distances / np.sum(distances)
        chosen_centroids_idx.append(np.random.choice(n, p=probabilities))
        chosen_centroids.append(data_points[chosen_centroids_idx[-1]])
    
    return np.array(chosen_centroids), chosen_centroids_idx

def main():
    k, max_iteration, eps, file_name_1, file_name_2 = validate_args(sys.argv)

    data_points = read_and_merge_files(file_name_1, file_name_2)

    try:
        if not data_points.shape[0] < k:
            raise ValueError
    except ValueError:
        exit_with_error(CLUSTERS_ERROR_MESSAGE)

    initial_centroids, initial_centroids_idx = init_centroids(k, data_points)
    
    result = fit(k, max_iteration, eps, data_points.tolist(), initial_centroids.tolist())

    print(','.join(map(str, initial_centroids_idx)))

    for centroid in result:
        print(','.join(f'{coord:.4f}' for coord in centroid))
        
    sys.exit(0)

if __name__ == "__main__":
    main()

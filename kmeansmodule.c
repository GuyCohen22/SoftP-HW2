#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct coordinate {
    double value;
    struct coordinate *next;
} Coordinate;

typedef struct vector
{
    struct vector *next;
    Coordinate *coordinates;
} Vector;

/* Frees the entire memory allocated to the matrix */
void free_matrix(Vector *head_vector) {
    Vector *vector_to_free;
    Coordinate *curr_coordinate, *coordinate_to_free;

    while (head_vector != NULL) {
        curr_coordinate = head_vector->coordinates;

        while (curr_coordinate != NULL) {
            coordinate_to_free = curr_coordinate;
            curr_coordinate = curr_coordinate->next;
            free(coordinate_to_free);
        }

        vector_to_free = head_vector;
        head_vector = head_vector->next;
        free(vector_to_free);
    }
}

/* Frees all memory allocated for a coordinate list starting from the given head */
void free_curr_coordinates(Coordinate *head_coordinate) {
    Coordinate *coordinate_to_free;

    while (head_coordinate != NULL) {
        coordinate_to_free = head_coordinate;
        head_coordinate = head_coordinate->next;
        free(coordinate_to_free);
    }
}

/* Allocates and returns a linked list of k centroids, each with the given dimension. returns NULL on failure. */
Vector *init_centroids_matrix(int k, int dimension) {
    Vector *head_vector, *curr_vector;
    Coordinate *head_coordinate, *curr_coordinate;
    int i, j;

    head_vector = calloc(1, sizeof(Vector));
    if (head_vector == NULL) {
        return NULL;
    }
    curr_vector = head_vector;

    for (i = 0; i < k; i++) {
        head_coordinate = calloc(1, sizeof(Coordinate));
        if (head_coordinate == NULL) {
            free_matrix(head_vector);
            return NULL;
        }
        curr_coordinate = head_coordinate;

        for (j = 1; j < dimension; j++) {
            curr_coordinate->next = calloc(1, sizeof(Coordinate));
            if (curr_coordinate->next == NULL) {
                free_matrix(head_vector);
                free_curr_coordinates(head_coordinate);
                return NULL;
            }
            curr_coordinate = curr_coordinate->next;
        }

        curr_vector->coordinates = head_coordinate;

        if (i != k - 1) {
            curr_vector->next = calloc(1, sizeof(Vector));
            if (curr_vector->next == NULL) {
                free_matrix(head_vector);
                return NULL;
            }
            curr_vector = curr_vector->next;
        }
    }
    return head_vector;
}

/*  Returns the Euclidean distance between two vectors */
double calculate_euclidean_distance(Vector *v1, Vector *v2) {
    double sum = 0.0, diff;
    Coordinate *c1 = v1->coordinates;
    Coordinate *c2 = v2->coordinates;
    
    while (c1 != NULL && c2 != NULL) {
        diff = c1->value - c2->value;
        sum += diff * diff;
        c1 = c1->next;
        c2 = c2->next;
    }
    
    return sqrt(sum);
}

/* Sets all values in the centroids matrix to 0 */
void zero_out_centroids(Vector *centroids, int k, int dimension) {
    Vector *curr_vector = centroids;
    Coordinate *curr_coordinate;
    int i, j;

    for (i = 0; i < k; i++) {
        curr_coordinate = curr_vector->coordinates;
        for (j = 0; j < dimension; j++) {
            curr_coordinate->value = 0.0;
            curr_coordinate = curr_coordinate->next;
        }
        curr_vector = curr_vector->next;
    }
}

/* Returns 1 if the centroids have converged, 0 otherwise */
int has_converged(Vector *prev_centroids, Vector *curr_centroids, int k, double eps) {
    Vector *local_prev_centroids = prev_centroids, *local_curr_centroids = curr_centroids;
    double curr_distance;
    int i;

     for (i = 0; i < k; i++) {
        curr_distance = calculate_euclidean_distance(local_prev_centroids, local_curr_centroids);
        if (curr_distance >= eps) return 0;
        local_prev_centroids = local_prev_centroids->next;
        local_curr_centroids = local_curr_centroids->next;
    }

    return 1;
}

/* Assigns a datapoint to the closest cluster and returns the index of that cluster */
int assign_datapoint_to_closest_cluster(Vector *point_to_assign, Vector *centroids, int k) {
    Vector *curr_centroid = centroids;
    int min_index = 0;
    double curr_distance;
    double min_distance = calculate_euclidean_distance(point_to_assign, curr_centroid);
    int i;

    for (i = 1; i < k; i++) {
        curr_centroid = curr_centroid->next;
        curr_distance = calculate_euclidean_distance(point_to_assign, curr_centroid);
        if (curr_distance < min_distance) {
            min_distance = curr_distance;
            min_index = i;
        }
    }

    return min_index;
}

/* Updates the current centroids based on the assignments and counts of datapoints in each cluster */
void update_curr_centroids(Vector *prev_centroids, Vector *curr_centroids, Vector *datapoints, int *assignments, int *counts, int k, int num_vectors, int dimension) {
    Vector *curr_datapoint = datapoints;
    Vector *curr_centroid, *prev_centroid;
    Coordinate *curr_datapoint_coordinate, *curr_centroid_coordinate, *prev_centroid_coordinate;
    int i, j, m;

    for (i = 0; i < num_vectors; i++) {
        curr_centroid = curr_centroids;
        for (j = 0; j < assignments[i]; j++) {
            curr_centroid = curr_centroid->next;
        }

        curr_datapoint_coordinate = curr_datapoint->coordinates;
        curr_centroid_coordinate = curr_centroid->coordinates;

        for (m = 0; m < dimension; m++) {
            curr_centroid_coordinate->value += curr_datapoint_coordinate->value;
            curr_centroid_coordinate = curr_centroid_coordinate->next;
            curr_datapoint_coordinate = curr_datapoint_coordinate->next;
        }
        curr_datapoint = curr_datapoint->next;
    }

    prev_centroid = prev_centroids;
    curr_centroid = curr_centroids;
    for (i = 0; i < k; i++) {
        prev_centroid_coordinate = prev_centroid->coordinates;
        curr_centroid_coordinate = curr_centroid->coordinates;
        
        for (j = 0; j < dimension; j++) {
            curr_centroid_coordinate->value = counts[i] > 0 ? (curr_centroid_coordinate->value) / counts[i] : prev_centroid_coordinate->value;
            prev_centroid_coordinate = prev_centroid_coordinate->next;
            curr_centroid_coordinate = curr_centroid_coordinate->next;
        }

        prev_centroid = prev_centroid->next;
        curr_centroid = curr_centroid->next;
    }
}

/* Returns centroids calculated using the k-means algorithm */
Vector *calculate_centroids_using_kmeans(int k, int maximum_iteration, int num_vectors, int dimension, Vector *datapoints, Vector *inital_centroids, Vector *curr_centroids, int *assignments, int*counts, double eps) {

    Vector *temp_centroids, *prev_centroids = inital_centroids, *local_curr_centroids = curr_centroids, *point_to_assign;
    int converged_flag = 0, iteration_cnt = 0;
    int i;

    while (iteration_cnt < maximum_iteration && !converged_flag) {
        memset(assignments, 0, sizeof(int) * num_vectors);
        memset(counts, 0, sizeof(int) * k);
        zero_out_centroids(local_curr_centroids, k, dimension);
        point_to_assign = datapoints;

        /* Assign each datapoint to the closest cluster */
        for (i = 0; i < num_vectors; i++) {
            assignments[i] = assign_datapoint_to_closest_cluster(point_to_assign, prev_centroids, k);
            counts[assignments[i]]++;
            point_to_assign = point_to_assign->next;
        }

        /* Calculate new centroids */
        update_curr_centroids(prev_centroids, local_curr_centroids, datapoints, assignments, counts, k, num_vectors, dimension);
        converged_flag = has_converged(prev_centroids, local_curr_centroids, k, eps);
        iteration_cnt++;
        if (!converged_flag) {
            /* Swap the previous and current centroids for the next iteration */
            temp_centroids = prev_centroids;
            prev_centroids = local_curr_centroids;
            local_curr_centroids = temp_centroids;
        }
    }
    
    return converged_flag ? local_curr_centroids : prev_centroids;
}

/* Converts a Python list of lists into a linked list matrix of Vectors */
Vector *pylist_to_vector_matrix(PyObject *data_points) {
    Vector *head_vector = NULL, *curr_vector = NULL;
    Coordinate *head_coordinate = NULL, *curr_coordinate = NULL;
    int num_vectors, dimension, i, j;
    PyObject *vec, *value;

    num_vectors = PyList_Size(data_points);

    for (i = 0; i < num_vectors; i++) {
        vec = PyList_GetItem(data_points, i);

        head_coordinate = calloc(1, sizeof(Coordinate));
        if (head_coordinate == NULL) {
            PyErr_NoMemory();
            free_matrix(head_vector);
            return NULL;
        }
        curr_coordinate = head_coordinate;

        dimension = PyList_Size(vec);

        for (j = 0; j < dimension; j++) {
            value = PyList_GetItem(vec, j);

            curr_coordinate->value = PyFloat_AsDouble(value);

            if (j < dimension - 1) {
                curr_coordinate->next = calloc(1, sizeof(Coordinate));
                if (curr_coordinate->next == NULL) {
                    PyErr_NoMemory();
                    free_curr_coordinates(head_coordinate);
                    free_matrix(head_vector);
                    return NULL;
                }
                curr_coordinate = curr_coordinate->next;
            }
        }

        if (head_vector == NULL) {
            head_vector = calloc(1, sizeof(Vector));
            if (head_vector == NULL) {
                PyErr_NoMemory();
                free_curr_coordinates(head_coordinate);
                free_matrix(head_vector);
                return NULL;
            }

            head_vector->coordinates = head_coordinate;
            curr_vector = head_vector;
        } 
        
        else {
            curr_vector->next = calloc(1, sizeof(Vector));
            if (curr_vector->next == NULL) {
                    PyErr_NoMemory();
                    free_curr_coordinates(head_coordinate);
                    free_matrix(head_vector);
                    return NULL;
                }
                curr_vector = curr_vector->next;
                curr_vector->coordinates = head_coordinate;
        }
    }
    return head_vector;
}

/* Converts a linked list matrix of Vectors into a Python list of lists */
PyObject *vector_matrix_to_pylist(Vector *head_vector) {
    PyObject *pylist_data_points = PyList_New(0);
    PyObject *vec;
    PyObject *val;
    Vector *curr_vector = head_vector;
    Coordinate *curr_coordinate;
    

    while (curr_vector != NULL) {
        vec = PyList_New(0);
        curr_coordinate = curr_vector->coordinates;

        while (curr_coordinate != NULL) {
            val = PyFloat_FromDouble(curr_coordinate->value);
            PyList_Append(vec, val);
            curr_coordinate = curr_coordinate->next;
        }

        PyList_Append(pylist_data_points, vec);
        curr_vector = curr_vector->next;
    }

    return pylist_data_points;
}

/* Parses Python arguments, runs the K-means++ algorithm, and returns final centroids as a Python list */
static PyObject *fit(PyObject *self, PyObject *args) {
    int k, maximum_iteration;
    int num_vectors, dimension;
    double eps;
    Vector *datapoints, *inital_centroids, *curr_centroids, *result_centroids;
    int *assignments, *counts;
    PyObject *data_points_object, *inital_centroids_object, *py_result;

    if (!PyArg_ParseTuple(args, "iidOO", &k, &maximum_iteration, &eps, &data_points_object, &inital_centroids_object)) {
        return NULL;
    }

    num_vectors = PyList_Size(data_points_object);
    dimension = PyList_Size(PyList_GetItem(data_points_object, 0));
    datapoints = pylist_to_vector_matrix(data_points_object);
    inital_centroids = pylist_to_vector_matrix(inital_centroids_object);

    curr_centroids = init_centroids_matrix(k, dimension);
    assignments = calloc(num_vectors, sizeof(int));
    counts = calloc(k, sizeof(int));
    if (inital_centroids == NULL || curr_centroids == NULL || assignments == NULL || counts == NULL) {
        PyErr_NoMemory();
        free_matrix(datapoints);
        free_matrix(inital_centroids);
        free_matrix(curr_centroids);
        free(assignments);
        free(counts);
        return NULL;
    }

    result_centroids = calculate_centroids_using_kmeans(k, maximum_iteration, num_vectors, dimension, datapoints, inital_centroids, curr_centroids, assignments, counts, eps);
    py_result = vector_matrix_to_pylist(result_centroids);

    free_matrix(datapoints);
    free_matrix(inital_centroids);
    free_matrix(curr_centroids);
    free(assignments);
    free(counts);

    return py_result;
}

static PyMethodDef kmeansppMethods[] = {
    {"fit",
    (PyCFunction) fit,
    METH_VARARGS,
    PyDoc_STR("fit(k, max_iter, epsilon, data_points, initial_centroids) -> centroids\n\n"
               "Runs the K-means++ clustering algorithm.\n\n"
               "Parameters:\n"
               "  k (int): Number of required clusters. Must satisfy 1 < k < N, where N is the number of data points.\n"
               "  max_iter (int): Optional. Maximum number of iterations. Must satisfy 1 < max_iter < 1000. Default is 300.\n"
               "  epsilon (float): Convergence threshold. Must be epsilon >= 0.\n"
               "  data_points (list of list of floats): N data points of dimension d.\n"
               "  initial_centroids (list of list of floats): List of k initial centroids.\n\n"
               "Returns:\n"
               "  list of list of floats: Final k centroids after convergence or reaching max_iter.")},
               {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mykmeansppmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanspp",     
    "C extension module for running the K-means++ clustering algorithm.",  
    -1,               
    kmeansppMethods   
};

PyMODINIT_FUNC PyInit_mykmeanspp(void) {
    PyObject *m;
    m = PyModule_Create(&mykmeansppmodule);
    if (!m) {
        return NULL;
    }
    return m;
}
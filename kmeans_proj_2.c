#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int get_closest_cluster_index(double *v, double **centroids, int d, int K);
static void assign_vector_to_cluster (double* v, double **centroids, double ***clusters, int *cluster_size, int d, int K);
static void update_centroid_value(double* centroid ,double ** cluster, int cluster_size, int d);
static int update_centroids_and_check_distance_difference(double **centroids, double*** clusters, int* cluster_sizes, int k, int d, double eps);
static double cal_norm (double *v1, double *v2, int d);
static int get_closest_cluster_index(double *v, double **centroids, int d, int K);
static void assign_vector_to_cluster (double* v, double **centroids, double ***clusters, int *cluster_size, int d, int K);
static int K_means(double **vectors_array, double **centroids_array, int N, int d, int K, int max_iter, double eps);
static void other_error();
static PyObject *fit(PyObject *self, PyObject *args);


static PyMethodDef KMeansMethods[] = {
    {"fit", (PyCFunction) fit, METH_VARARGS, PyDoc_STR("Python interface for mykmeanssp C library function")},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    "Python interface for the fputs C library function",
    -1,
    KMeansMethods
};

static PyObject *fit(PyObject *self, PyObject *args) {
	PyObject* vectors_obj, *centroids_obj;
	double **vectors_array,  **centroids_array;
	double *vectors, *centroids;
	int N, d, k, max_iter;
	double eps;
    int error;

    int i, j;
    PyObject *item;
    PyObject *list;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "OOiiiid", &vectors_obj, &centroids_obj, &N, &d, &k, &max_iter, &eps)) {
        return NULL;
    }

	/*allocating space for vectors and centroids */

	vectors = (double*)calloc(N*d, sizeof(double));
	vectors_array = (double**)calloc(N, sizeof(double *));

	centroids = (double*)calloc(k*d, sizeof(double));
	centroids_array = (double**)calloc(k, sizeof(double *));

	if(vectors == NULL || vectors_array == NULL || centroids == NULL || centroids_array == NULL){
    	free(vectors);
    	free(vectors_array);
		free(centroids);
	    free(centroids_array);
    	other_error();
    }

	/* create matrices for vectors and centroids */
	for (i = 0 ; i < N ; i++) {
		vectors_array[i] = vectors + i*d;
	}

	for (i = 0 ; i < k ; i++) {
	    centroids_array[i] = centroids + i*d;
	}

    /* python obj to double** */

       for (i = 0; i < N; i++) {
    	   list = PyList_GetItem(vectors_obj, i);
    	   	for (j = 0; j < d; j++) {
    	   		item = PyList_GetItem(list, j);
    	   		if (!PyFloat_Check(item)){
    	   			vectors_array[i][j] = 0.0; /* or error? */
				   }
    	   		vectors_array[i][j] = PyFloat_AsDouble(item);
       }
    }

       for (i = 0; i < k; i++) {
		    list = PyList_GetItem(centroids_obj, i);
     	   	for (j = 0; j < d; j++) {
            	item = PyList_GetItem(list, j);
            	if (!PyFloat_Check(item)){
					centroids_array[i][j] = 0.0; /* or error? */
				}
            	centroids_array[i][j] = PyFloat_AsDouble(item);
        	}
     	}

    error = K_means(vectors_array, centroids_array, N, d, k, max_iter, eps);

    free(vectors);
    free(vectors_array);

    if(error == 1){
    	free(centroids);
    	free(centroids_array);
    	other_error();
    }


	/* translating the double** centroids array to a python list [[]] */
    list = PyList_New(k);
    if(list == NULL) {
    	free(list);
		free(centroids);
    	free(centroids_array);
        other_error();
    }

    for(i = 0; i < k; i++) {
        item = PyList_New(d);
        if(item == NULL) {
        	error = 1;
        	break;
        }

        for(j = 0; j < d; j++) {
            PyList_SET_ITEM(item, j,  PyFloat_FromDouble(centroids_array[i][j]));
        }

        PyList_SET_ITEM(list, i, item);
    }

    free(centroids);
    free(centroids_array);

    if(error == 1){
    	free(item);
		free (list);
        other_error();
    }

    return list;
}

PyMODINIT_FUNC PyInit_mykmeanssp(void) {
	PyObject *m;

    m = PyModule_Create(&kmeansmodule);
    if(!m){
    	return NULL;
    }
    return m;
}

static int K_means(double **vectors_array, double **centroids_array, int N, int d, int K, int max_iter, double eps){

	int i, iteration_number, flag;
	double **clusters;
	double ***clusters_array;
	int *clusters_sizes;

    /*allocate space for clusters. each cluster is an array of pointers
    (that is, clusters_array in the [i][j] contains pointer to a vector)*/
    clusters = (double **)calloc(K, N*(sizeof(double *)));
    clusters_array = (double***)calloc(K, sizeof(double **));
    /*allocate spaece for clusters sizes*/
    clusters_sizes = (int*)calloc(K, sizeof(int));

    if(clusters == NULL || clusters_array == NULL || clusters_sizes == NULL){
    	free(clusters);
    	free(clusters_array);
    	free(clusters_sizes);
    	return 1;
    }

    for (i = 0 ; i < K ; i++) { /*make K arrays, each for one cluster*/
        clusters_array[i] = clusters + i*N;
    }

    /*repeat until convergence or until iteration number = max iter)*/
    iteration_number = 0;
    while (iteration_number < max_iter) {

		/*reser cluster sizes*/
		for (i = 0 ; i < K ; i++){
    		clusters_sizes[i] = 0;
    	}

		/*for each vector (N in total), assign vector to the closest cluster*/
        for (i = 0 ; i < N ; i++){
            /*the vector we want is vectors_array[i]*/
            assign_vector_to_cluster (vectors_array[i], centroids_array, clusters_array, clusters_sizes, d, K);
        }

        /*for each centroid (K in total), update centroid, and check if the change is < EPSILON*/
        flag = update_centroids_and_check_distance_difference(centroids_array, clusters_array, clusters_sizes, K, d, eps);

        /*if there was an error in update_centroids function*/
        if(flag <0){
        	free(clusters);
        	free(clusters_array);
        	free(clusters_sizes);
        	return 1;
        }

        /*check convergence
        flag == 0 iff all changes are smaller then EPSILON, that is, there is convergence*/
        if (flag == 0){
            break;
        }

        iteration_number++;
    }

    free(clusters);
    free(clusters_array);
    free(clusters_sizes);

    return 0;
 }


/*calculate Euclidean distance of 2 vectors, v1 and v2, both of dimension d*/
static double cal_norm (double *v1, double *v2, int d) {

	int i;
	double sum = 0, norm;
	for (i=0; i<d; i++){
		sum += pow ( (*(v1+i)) - (*(v2+i)),2 );
	}

	norm = sqrt (sum);
	return norm;
}

static void other_error() {
	printf("An Error Has Occurred\n");
	exit(1);
}



static int get_closest_cluster_index(double *v, double **centroids, int d, int K) {
	int i;
	int closest_cluster_index = -1;
	double min_distance, distance;
	for (i=0; i<K; i++){
		distance = cal_norm(v, (*(centroids+i)), d); /*the centroid we check is *(centorids+i)*/
		if (closest_cluster_index < 0 || distance < min_distance) {
			min_distance = distance;
			closest_cluster_index = i;
		}
	}
	return closest_cluster_index;
}


static void assign_vector_to_cluster (double* v, double **centroids, double ***clusters, int *cluster_size, int d, int K){
	int cluster_index, new_vector_index;

	/*find the closest cluster*/
	cluster_index = get_closest_cluster_index(v, centroids,d, K);

    		/*check how many vectors already in cluster, to add v in the right place
    		num_of_vectors_in_cluster is *(cluster_size+i)
    		if we had x vectors, places 0, 1, ... , x-1 are occupid, and new vector goes to place x*/
    new_vector_index = cluster_size[cluster_index];

	/*assign the vector
	our cluster is = *(clusters + cluster_index)
	the place to put the vector is = *(cluster + new_vector_index)*/
	clusters[cluster_index][new_vector_index] = v;

	/*update cluster size*/
	cluster_size[cluster_index] ++;

}

static void update_centroid_value(double* centroid ,double ** cluster, int cluster_size, int d){
	int i,j;

	/*initialize the centroid*/
	for(i=0;i<d;i++) {
		centroid[i] = 0;
	}

	if(cluster_size == 0) {
		return;
	}

	/*calculate new value for centroid*/
	for(i=0;i<cluster_size;i++) {
		for(j=0;j<d;j++) {
			centroid[j] +=cluster[i][j];
		}
	}
	for(i=0;i<d;i++) {
		centroid[i] /= cluster_size;
	}
}



static int update_centroids_and_check_distance_difference(double **centroids, double*** clusters, int* cluster_sizes, int k, int d, double eps){
	int flag, i, j;
	double norm;
	double *temp_vector;

	flag = 0;
	temp_vector = (double*)malloc(d*sizeof(double));

	if(temp_vector == NULL){
		free(temp_vector);
		return -1;
	}

	for(i=0;i<k;i++){ /*for every centroid, update*/
		for(j=0;j<d;j++){ /*save old value of centroid*/
			temp_vector[j] = centroids[i][j];
		}
		update_centroid_value(centroids[i], clusters[i], cluster_sizes[i], d);
		norm = cal_norm(temp_vector, centroids[i],d);
		if(norm >= eps){
			flag =1;
		}
	}
	free(temp_vector);
	return flag;
}


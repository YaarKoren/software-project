#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#define EPSILON 0.001
#define MAX_ITER_DEFAULT 200

int get_closest_cluster_index(double *v, double **centroids, int d, int K);
void assign_vector_to_cluster (double* v, double **centroids, double ***clusters, int *cluster_size, int d, int K);
int read_from_file(FILE *ifp, double** vectors, int N, int d);
int write_to_file(char* path, double **array, int d, int k);
int get_N(FILE *ifp);
int get_d(FILE *ifp);
void update_centroid_value(double* centroid ,double ** cluster, int cluster_size, int d);
int update_centroids_and_check_distance_difference(double **centroids, double*** clusters, int* cluster_sizes, int k, int d);
double cal_norm (double *v1, double *v2, int d);
int get_closest_cluster_index(double *v, double **centroids, int d, int K);
void assign_vector_to_cluster (double* v, double **centroids, double ***clusters, int *cluster_size, int d, int K);
int K_means(double **vectors_array, double **centroids_array, int N, int d, int K, int max_iter);
void other_error();
void input_error();

int main(int argc, char **argv){
	int i, N, d, k, max_iter, error;
	char * input_path, *output_path;
	double *vectors;
	double ** vectors_array;
	double *centroids;
    double **centroids_array;
	FILE *ifp;

	k = atoi(argv[1]); /*if not a valid int, returns 0 */

	if(argc == 4){ /*max_iter not provided name of command + 3 args*/
		max_iter = MAX_ITER_DEFAULT;
		input_path = argv[2];
		output_path = argv[3];
	} else {
		if(argc == 5) { /*max_iter provided - name of command + 4 args*/
			max_iter = atoi (argv[2]); /*if not a valid int, returns 0*/
			input_path = argv[3];
			output_path = argv[4];
		} else {
			input_error();
		}
	}

	/*open file*/
	ifp = fopen(input_path, "r");
	if (ifp == NULL){  
		fclose(ifp);
		input_error();
	}

	N = get_N(ifp);
	rewind(ifp);
	d = get_d(ifp);
	rewind(ifp);

	if(!(k>1 && k<N) || max_iter <=0){
		fclose(ifp);
		input_error();
	}

	/*create an N*d matrix to store the vectors*/
	vectors = (double*)calloc(N*d, sizeof(double));
	vectors_array = (double**)calloc(N, sizeof(double *));

	if(vectors == NULL || vectors_array == NULL){
    	free(vectors);
    	free(vectors_array);
    	other_error();
    }

	for (i = 0 ; i < N ; i++) {
		vectors_array[i] = vectors + i*d;
	}

	/*read from file and store the vectors*/
	error = read_from_file(ifp, vectors_array, N, d);
	fclose(ifp);
	if(error) {
		free(vectors);
		free(vectors_array);
		other_error();
	}

    /*initialize k centroids as first K vectors
    create an k*d matrix to store the centroids*/
    centroids = (double*)calloc(k*d, sizeof(double));
    centroids_array = (double**)calloc(k, sizeof(double *));

    if(centroids == NULL || centroids_array == NULL){
    	free(centroids);
    	free(centroids_array);
		free(vectors);
    	free(vectors_array);
    	other_error();
    }

    for (i = 0 ; i < k ; i++) {
         centroids_array[i] = centroids + i*d;
    }

	error = K_means(vectors_array, centroids_array, N, d, k, max_iter);
	if(error == 1){
		free(centroids);
		free(centroids_array);
		free(vectors_array);
		other_error();
	}

	error = write_to_file(output_path, centroids_array, d, k);

	free(vectors);
	free(centroids_array);
	free(centroids);
	free(vectors_array);

	if(error){
		input_error(); /*after a discussion we decided it is an input error, the file name was not leagal for some reason*/
	}
	exit(0);
}

int K_means(double **vectors_array, double **centroids_array, int N, int d, int K, int max_iter){

	 int i, j, iteration_number, flag;
	 double **clusters;
	 double ***clusters_array;
	 int *clusters_sizes;
    /*store the k first vectors as the k centroids*/
     for  (i = 0; i < K; i++) { /* run K times - number of centroids*/
        for (j = 0; j < d; j++) { /* run d times - number of coordinates*/
            centroids_array[i][j] = vectors_array[i][j];
        }
     }

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
        flag = update_centroids_and_check_distance_difference(centroids_array, clusters_array, clusters_sizes, K, d);

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
double cal_norm (double *v1, double *v2, int d) {

	int i;
	double sum = 0, norm;
	for (i=0; i<d; i++){
		sum += pow ( (*(v1+i)) - (*(v2+i)),2 );
	}

	norm = sqrt (sum);
	return norm;
}

void input_error(){
	printf("Invalid Input!\n");
	exit(1);

}

void other_error() {
	printf("An Error Has Occurred\n");
	exit(1);
}

/*get number of vectors*/
int get_N(FILE *ifp) {
	int count;
	char c;

	count = 0;
	for (c = fgetc(ifp); c != EOF; c = fgetc(ifp))
		if (c == '\n') /*newline*/
			count++;

	/*last line is always empty, but in case of 4 lines there will be exactly 4 '\n's*/
	return count;

}

/*get dimension (number of coordinates per vector)*/
int get_d(FILE *ifp) {
	int count;
	char c;

	count = 0;
	while  ( (c = fgetc(ifp) ) != '\n') {
		if (c == ',')
			count++;
	}

	count++; /*because the line ends with "\n" not ","*/
	return count;

}


int read_from_file(FILE *ifp, double **vectors, int N, int d) {
	int i, j;
	char c;
	char *start_of_str_coor, *str_coor;
	double coor;

	/*store the vectors*/
	str_coor = (char *)malloc(7 * sizeof(char)); /* 7 becausse longest num i.e.: "-7.8602"*/
	if (str_coor == NULL){
		free(str_coor);
		return 1;
	}
	for  (i = 0; i < N; i++) { /* run N times - number of vectors = number of lines (minus the empty one)*/
		for (j = 0; j < d; j++) { /* run d times - number of coordinates = number of numbers in a line*/
			start_of_str_coor = str_coor;
			while ( (c=fgetc(ifp)) != ',' && c != '\n') { /*collect the chars of the coordinate*/
				*str_coor = (char)c;
				str_coor++;
			}
			coor = strtod(start_of_str_coor, NULL);
			vectors[i][j] = coor;
			str_coor = start_of_str_coor;
		}

	}
	free(str_coor);
	return 0;
}


int get_closest_cluster_index(double *v, double **centroids, int d, int K) {
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


void assign_vector_to_cluster (double* v, double **centroids, double ***clusters, int *cluster_size, int d, int K){
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

void update_centroid_value(double* centroid ,double ** cluster, int cluster_size, int d){
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



int update_centroids_and_check_distance_difference(double **centroids, double*** clusters, int* cluster_sizes, int k, int d){
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
		if(norm >= EPSILON){
			flag =1;
		}
	}
	free(temp_vector);
	return flag;
}

int write_to_file(char* path, double **array, int d, int k){
	FILE * of;
	int i, j;

	of = fopen(path,"w");
	if (of == NULL){
			return 1;
	}

	for(i=0;i<k;i++){
		for(j=0;j<d;j++){
			if(j==d-1){
				fprintf(of,"%.4f", array[i][j]);
			} else {
				fprintf(of,"%.4f,", array[i][j]);
			}

		}
		fprintf(of,"\n");
	}
	fclose(of);
	return 0;
}

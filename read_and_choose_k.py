import numpy as np
import sys
np.random.seed(0)  

def main(args):
    #args = [k, max_iter (optional), eps,filename1,filename2]
    if len(args) != 4 and len(args) != 5:
        invalid_input_error()
    arg_index = 1
    try:
        k = int(args[0])
        if len(args) == 5:
            max_iter = int(args[1] if args[1] else 300)
            arg_index += 1
        else:
            max_iter = 300
    except ValueError:
        invalid_input_error()
    eps = args[arg_index]
    input_path_1 = args[arg_index + 1]
    input_path_2 = args[arg_index + 2]
    if max_iter <= 0 or k < 1:
        invalid_input_error()
    if not ((input_path_1.endswith(".txt") or input_path_1.endswith(".csv"))
            and (input_path_2.endswith(".txt") or input_path_2.endswith(".csv"))):
        invalid_input_error()
    vectors_dict_1 = read_from_file(input_path_1)
    vectors_dict_2 = read_from_file(input_path_2)
    vectors = [[0.0] * (len(vectors_dict_1[1.0]) + len(vectors_dict_2[1.0])) for i in range(len(vectors_dict_1))]
    for index_key in vectors_dict_1:
        vectors[int(index_key)] = vectors_dict_1[index_key] + vectors_dict_2[index_key]
    # usable from here: vectors = [[]], k = int, max_iter = int, eps
    
    d = len(vectors[0])
    centroids = np.zeros((k, d))

    res_indices = k_means_pp(vectors, centroids, k)

    #call the k-means algoritm with centroids

    #print the chosen vector indices (k in total) for the initialized centroids
    print(*res_indices, sep = ", ") 

    #print the k final centoris
    for i in range(k):
        print(*centroids[i], sep = ", ") 


def read_from_file(path):
    vectors_dict = {}
    try:
        lines_in_file = open(path, 'r').readlines()
    except FileNotFoundError:
        invalid_input_error()
    except:
        general_error()
    for line in lines_in_file:
        if not line:
            continue
        a = line.split(',')
        values = [float(strval) for strval in a]
        vectors_dict[values[0]] = values[1:] # map index to values. Is the num necessarily int? cast to int?
    return vectors_dict


def invalid_input_error():
    print("Invalid Input!")
    exit(1)


def general_error():
    print("An Error Has Occurred")
    exit(1)


def k_means_pp(vectors, centroids, k):
    # vectors - a matrix N x d, of N vectors, each one d coordinates. from the input files
    # centroids - a matrix k x d, initialized with zeros, to store the k centroids
    # the function initalize k centroids, out of the N vectors, according to the kmeans++ algotithm
    # the function returns a list of indices of the chosen vectors

    N = len(vectors)
    d = len(vectors[0])
    i = 0
    indices = np.arange(N)  # N vectors: 0,1, ... , N-1

    # initialize k-size lists to store the chosen vectors indices
    chosen_indices = [None] * k

    # select first centroid randomly from the N vectors
    rand_index = np.random.choice(indices)
    centroids[i] = np.copy(vectors[rand_index])  # create copy, to prevent changes in the vectors matrix
    chosen_indices[i] = rand_index

    # initialize N-size lists to store D values  and probabilty values for each vector
    D_lst = [None] * N
    P_lst = [None] * N

    while i < k-1:
        for j in range(N):
            vector = np.copy(vectors[j])
            D_vector = calculate_D(vector, centroids, i)
            D_lst[j] = D_vector

        D_sum = sum(D_lst)

        for j in range(N):
            P_lst[j] = D_lst[j] / D_sum

        i += 1

        # choose the vector randomlly, with the probabilty calculated
        rand_index = np.random.choice(indices, p = P_lst)
        centroids[i] = np.copy(vectors[rand_index])  # create copy, to prevent changes in the vectors matrix
        chosen_indices[i] = rand_index

    return chosen_indices


def calculate_D(vector, centroids, i):
    min_value = -1
    for j in range(i + 1):
        norm = cal_norm_squared(vector, centroids[j])
        if (min_value == -1 or norm < min_value):
            min_value = norm
    return min_value


def cal_norm_squared(v1, v2):
    args_sum = 0
    for i in range(len(v1)):
        args_sum += ((v1[i] - v2[i]) ** 2)
    return args_sum

if __name__ == "__main__":
    main(sys.argv[1:])
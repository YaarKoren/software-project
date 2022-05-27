import mykmeanssp
import pandas as pd
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
        eps = float(args[arg_index])
    except ValueError:
        invalid_input_error()
    input_path_1 = args[arg_index + 1]
    input_path_2 = args[arg_index + 2]
    if max_iter <= 0 or k < 1:
        invalid_input_error()
    if not ((input_path_1.endswith(".txt") or input_path_1.endswith(".csv"))
            and (input_path_2.endswith(".txt") or input_path_2.endswith(".csv"))):
        invalid_input_error()

    vectors_1 = pd.read_csv(input_path_1, header=None) #make the first row in the data file a regular row in the df, and not columns names
    vectors_2 = pd.read_csv(input_path_2, header=None) 
    make_df_for_merge(vectors_1)
    make_df_for_merge(vectors_2)
    vectors = pd.merge(vectors_1, vectors_2 ,on="index", sort=True)
    vectors = vectors.set_index("index")
    N, d = vectors.shape
    col_names = generate_col_names_list(d)
    vectors.columns = col_names
    
    centroids = np.zeros((k, d))

    res_indices = k_means_pp(vectors, centroids, k)
    res_indices_int = [int(i) for i in res_indices]

    #convert centroids and vectors to lists, for the c module
    centroids_lst = centroids.tolist()
    vectors_lst = vectors.values.tolist()

    result_centroids = mykmeanssp.fit( vectors_lst, centroids_lst, N, d, k, max_iter, eps)
    
    # print the results: the chosen k indices in algorithm 1, and the k final centoris
    print(*res_indices_int, sep = ", ")
    for i in range(k):
        print(','.join("%0.4f" % x for x in result_centroids[i]))


def generate_col_names_list(col_num, start=0):
    names_list = []
    for i in range(start, (col_num+start)):
        names_list += ["point_"+str(i)]
    return names_list


def make_df_for_merge(df):
    col_num = df.shape[1] - 1 # columns number, minus 1 because the first column is the index
    col_names = ["index"] + generate_col_names_list(col_num)   
    df.columns = col_names


def invalid_input_error():
    print("Invalid Input!")
    exit(1)


def general_error():
    print("An Error Has Occurred")
    exit(1)



def k_means_pp(vectors, centroids, k):
    # vectors - a pandas DataFrame N x d, of N vectors, each one d coordinates. from the input files
    # centroids - a numpy matrix k x d, initialized with zeros, to store the k centroids
    # the function initalize k centroids, out of the N vectors, according to the kmeans++ algotithm
    # the function returns a list of indices of the chosen vectors

    N, d = vectors.shape
    i = 0
    indices = vectors.index.to_list()  

    # initialize k-size lists to store the chosen vectors indices
    chosen_indices = [None] * k

    # select first centroid randomly from the N vectors
    select_random_vector(vectors, centroids, indices, chosen_indices, i)

    # initialize N-size lists to store D values  and probabilty values for each vecto
    D_lst = [None] * N
    P_lst = [None] * N

    while i < k-1:
        for j in range(N):
            vector = vectors.iloc[j].to_list()
            D_vector = calculate_D(vector, centroids, i)
            D_lst[j] = D_vector

        D_sum = sum(D_lst)

        for j in range(N):
            P_lst[j] = D_lst[j] / D_sum

        i += 1

        # choose the vector randomly, vector in index [i] has P_lst[i] probablity to be chosen
        select_random_vector(vectors, centroids, indices, chosen_indices, i, probablity = P_lst)

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

def select_random_vector(vectors, centroids, indices, chosen_indices, i, probablity = None):
    # select centroid randomly from the N vectors
    if probablity == None: # each vector has the same probablity to be chosen
        rand_index = np.random.choice(indices)
    if probablity != None: # vector in index [i] has probablity[i] probablity to be chosen
        rand_index = np.random.choice(indices, p = probablity)
    centroids[i] = vectors.loc[rand_index].to_list() 
    chosen_indices[i] = rand_index

if __name__ == "__main__":
    main(sys.argv[1:])
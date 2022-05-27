import pandas as pd
import numpy as np
import sys

from kmeans_pp import invalid_input_error
np.random.seed(0)

def main(args):
    #args = [k, max_iter (optional), eps,filename1,filename2]
    print("running")
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
    vectors = files_to_vectors(input_path_1, input_path_2)
    print("finished reading from file")
    print(vectors.shape)
    N, d = vectors.shape
    print("N is: ", N)
    print("d is: ", d)

    # print("vectors[0]:")
    # print(vectors[0])

    print("vectors.index:")
    print(vectors.index)
    print("vectors indices to list")
    print(vectors.index.to_list())


def files_to_vectors(path_1, path_2):
    vectors_1 = pd.read_csv(path_1, header=None) #make the first row in the data file a regular row in the df, and not columns names
    vectors_2 = pd.read_csv(path_2, header=None) 
    print("vectors1:")
    print(vectors_1)
    print("vectors2:")
    print(vectors_2)




    dim_1 = vectors_1.shape[1] - 1 # minus 1 because the first column is the index
    dim_2 = vectors_1.shape[1] - 1 # minus 1 because the first column is the index
    print("dim1 is: ", dim_1)
    print("dim2 is: ", dim_2)

    col_names_1 = ["index"] + generate_col_names_list(dim_1)
    print(col_names_1)

    col_names_2 = ["index"] + generate_col_names_list(dim_2, start=dim_1)
    print(col_names_2)

    vectors_1.columns = col_names_1
    vectors_2.columns = col_names_2

    print("vectors1:")
    print(vectors_1)
    print("vectors2:")
    print(vectors_2)

    vectors = pd.merge(vectors_1, vectors_2,on="index")

    print("merged:")
    print(vectors)

    vectors = vectors.set_index("index")

    vectors.sort_index(inplace=True)

    print("sorted:")
    print(vectors)

    return vectors




def generate_col_names_list(col_num, start=0):
    names_list = []
    for i in range(start, (col_num+start)):
        names_list += ["point_"+str(i)]
        print(names_list)
    return names_list




def invalid_input_error():
    print("Invalid Input!")
    exit(1)


def general_error():
    print("An Error Has Occurred")
    exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
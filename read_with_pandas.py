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

    vectors_1 = pd.read_csv(input_path_1, header=None) #make the first row in the data file a regular row in the df, and not columns names
    vectors_2 = pd.read_csv(input_path_2, header=None) 
    make_df_for_merge(vectors_1)
    make_df_for_merge(vectors_2)
    vectors = pd.merge(vectors_1, vectors_2 ,on="index", sort=True)
    vectors = vectors.set_index("index")
    N, d = vectors.shape
    col_names = generate_col_names_list(d)
    vectors.columns = col_names

    print("vectors:")
    print(vectors)
    # print("vectors[0]:")
    # print(vectors[0])

    print("vectors.index:")
    print(vectors.index)
    print("vectors indices to list")
    print(vectors.index.to_list())

    print("first row by index:")
    print(vectors.iloc[0])

    print("point_0 by index- index 0:")
    print(vectors.loc[0.0,'point_0'])

    print("first row by index- index 0:")
    print(vectors.loc[0.0])

    vector = vectors.loc[0.0].to_list()
    print('vector is:')
    print(vector)

    vector[0] = 1.1111
    ("changed vector:")
    print(vector)
    ("vectors DataFrame:")
    print(vectors)

    vectors_mat = vectors.to_numpy
    print("vectors numpy matrix:")
    print(vectors_mat)
    #print("vectors numpy matrix shape:")
    #print(vectors_mat.shape)

    vectors_list = vectors.values.tolist()
    print("vectors list:")
    print(vectors_list)  

    first_col_list = vectors.iloc[:, 0].to_list()
    print("first_col_list:")
    print(first_col_list)


def generate_col_names_list(col_num, start=0):
    names_list = []
    for i in range(start, (col_num+start)):
        names_list += ["point_"+str(i)]
        print(names_list)
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


if __name__ == "__main__":
    main(sys.argv[1:])
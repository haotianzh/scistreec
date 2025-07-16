import popgen
from popgen.utils import ScisTree2
import numpy as np

import popgen.utils 
def read_scistree_input(file):
    mat = []
    with open(file, 'r') as f:
        line = f.readline()
        n, m = int(line.strip().split()[1]), int(line.strip().split()[2]) 
        for i in range(n):
            line = f.readline().strip()
            line = line.split()[1:]
            mat.append([float(_) for _ in line])
    return np.array(mat)


# prob_file = '../simulation/test1/1.prob'
# scistree2 = ScisTree2(threads=30)
# probs = read_scistree_input(prob_file)
# nwk, time = scistree2.infer(probs)

# print(nwk, time)






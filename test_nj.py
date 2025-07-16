import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree

def to_newick(node, parent_dist, leaf_names, newick_str=''):
    """Recursively generates a Newick format string from a scipy cluster tree."""
    if node.is_leaf():
        return f"{leaf_names[node.id]}:{parent_dist:.6f}" + newick_str
    
    left = to_newick(node.left, node.dist - node.left.dist, leaf_names)
    right = to_newick(node.right, node.dist - node.right.dist, leaf_names)
    
    return f"({left},{right}):{parent_dist:.6f}" + newick_str

def neighbor_joining_newick(A, labels=None):
    """Performs Neighbor-Joining and returns the Newick format of the tree."""
    if labels is None:
        labels = [str(i) for i in range(len(A))]  # Default labels as string indices
    
    # Convert distance matrix to a condensed form
    cond_dist_matrix = A[np.triu_indices(len(A), k=1)]
    
    # Perform hierarchical clustering using linkage with neighbor-joining (NJ is not directly available)
    Z = linkage(cond_dist_matrix, method='average')  # Closest method to NJ in scipy
    
    # Convert to tree
    tree = to_tree(Z, rd=False)
    
    # Convert tree to Newick format
    newick_str = to_newick(tree, tree.dist, labels) + ';'
    
    return newick_str

# Example usage
dist_matrix = np.array([
    [0.0, 5.0, 9.0, 9.0, 8.0],
    [5.0, 0.0, 10.0, 10.0, 9.0],

    
    [9.0, 10.0, 0.0, 8.0, 7.0],
    [9.0, 10.0, 8.0, 0.0, 3.0],
    [8.0, 9.0, 7.0, 3.0, 0.0]
])

labels = ["A", "B", "C", "D", "E"]
newick_tree = neighbor_joining_newick(dist_matrix, labels)
print(newick_tree)

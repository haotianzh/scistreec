import popgen
import numpy as np
import pandas as pd 


dirname = 'simulation/test1'
i = 1



def load_tree(dirname, i):
    tree_file = f'{dirname}/trees_dir/trees.{i:04}'
    with open(tree_file) as f:
        nwk = f.readline().strip()
        tree = popgen.utils.from_newick(nwk)
        tree = popgen.BaseTree(root=tree.root.get_children()[1]) # exclude outgroup
    return tree

def load_true_genotype(dirname, i):
    true_hap_file = f'{dirname}/true_haplotypes_dir/true_hap.{i:04}'
    haps = []
    cell_names = []
    with open(true_hap_file) as f:
        line = f.readline().strip()
        num_cells, num_sites = int(line.split()[0]), int(line.split()[1])
        num_cells = num_cells // 2 - 1 # exclude outgroup
        for i in range(num_cells):
            cell_name, hap_parental = f.readline().strip().split()
            cell_name, hap_maternal = f.readline().strip().split()
            hap = [f'{hap_parental[j]}{hap_maternal[j]}' for j in range(num_sites)]
            haps.append(hap)
            cell_names.append(cell_name[:-1])
    df = pd.DataFrame(index=cell_names, data=haps)
    # heter_columns = []
    # for i in range(num_sites):
    #     unique_gts = df[i].unique()
    #     if len(unique_gts) == 1 and unique_gts[0][0] == unique_gts[0][1]:
    #         print(unique_gts)
    #         continue
    #     else:
    #         heter_columns.append(i)
    # df = df[heter_columns]
    return df


def scan_for_deletion(genotype):
    sites = []
    for i in range(genotype.shape[1]):
        for gt in genotype[i].unique():
            if '-' in gt and i not in sites:
                sites.append(i)
    return sites 


def on_branch(leaves, cells):
    # if leaves == cells
    if len(leaves) != len(cells):
        return False
    for c in cells:
        if c not in leaves:
            return False
    return True


def find_deletion_on_tree(tree, geno, reads, del_site, verbose=False):
    # print('Delete site', del_site)
    DEL = 'DEL'
    OK = 'OK'
    tree = tree.copy()
    site = geno[del_site]
    read = reads[del_site]
    del_cells = site[site.apply(lambda x: '-' in x)]
    del_cells = del_cells.index.tolist()
    traversor = popgen.utils.TraversalGenerator()
    for node in traversor(tree):
        node.event = ''
        if node.is_leaf():
            node.event = f'{node.name} {node.event} [{site.loc[node.name]}] [{read[int(node.name[-4:])-1]}]'
        leaves = [n.name for n in node.get_leaves()]
        # print(leaves, del_cells)
        if on_branch(leaves, del_cells):
            node.event = f'{node.event} [{DEL}]'
            for des in node.get_descendants():
                des.event =  f'{des.event} [{DEL}]'
        # else:
        #     node.event = f'{node.event} [{OK}]'
    if verbose:
        tree.draw(nameattr='event')
    return tree
        
def read_vcf(vcf):
    order = ['A', 'C', 'G', 'T']
    # order: A, C, G, T
    data = []
    tg = []
    count = 0
    with open(vcf, 'r') as f:
        for line in f.readlines():
            data_site = []
            tg_site = []
            if line.startswith('##'):
                continue
            if line.startswith('#'):
                line = line.strip().split()
                cell_names = line[9: -1]
                continue
            line = line.strip().split()
            ref = line[3]
            for cell in line[9: -1]:
                info = cell.split(':')
                true_gt = info[-1]
                cn = 1 if '-' in true_gt else 2
                ml_gt = info[0]
                reads = [int(_) for _ in info[2].split(',')]
                ref_count = reads[order.index(ref)]
                alt_count = sum(reads) - ref_count
                # if '-' in true_gt:
                #     print(count, (ref_count, alt_count, cn))
                data_site.append((ref_count, alt_count, cn))
                tg_site.append(true_gt)
            data.append(data_site)
            tg.append(tg_site)
            count += 1
    
    return np.array(data, dtype=np.object_), pd.DataFrame(index=cell_names, data=np.array(tg, dtype=np.object_).T)


def get_scistreec_input_with_cn(dirname, i):
    vcf = f'{dirname}/vcf_dir/vcf.{i:04}'
    data, tg = read_vcf(vcf)
    tree = load_tree(dirname, i)
    return data, tree, tg


if __name__ == "__main__":
    # tree = load_tree(dirname, i)
    reads, tree, geno = get_scistreec_input_with_cn(dirname, i)
    print(reads.shape)
    # geno = load_true_genotype(dirname, i)
    del_sites = scan_for_deletion(geno)
    # print(del_sites)
    # for del_site in del_sites:
    del_tree = find_deletion_on_tree(tree, geno, reads, 626, verbose=True)
    
    # print(reads[626])

    # print(tree)

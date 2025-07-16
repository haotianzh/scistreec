import os 
import sys
import numpy as np
from datetime import date
import multiprocessing as mp 
import time 

class SimulatedVCF:
    def __init__(self, vcf_file):
        self.vcf_file = vcf_file
        self.base2num = {'A':0, 'C':1, 'G':2, 'T':3}
        start = time.time()
        self.load()
        # print(time.time()-start)
        # print('finish loading')
        
        self.get_PL()
        # self.get_P_10gt()
        self.get_P_10gt_2()
        # self.get_P_10gt_indv()
        # print(np.array(self.probs))

    def load(self):
        self.nsite = 0
        self.ncell = 0
        self.refs = []
        self.alts = []
        self.afs = []
        self.gls = []
        self.gts = []
        self.tgs = []
        self.mlgs = []
        self.reads = []
        self.gl10s = []

        vcf = open(self.vcf_file, 'r')
        for line in vcf.readlines():
            if line.startswith("#"):
                continue
            self.nsite += 1
            self.ncell = len(line.strip().split()[9: -1])
            line = line.strip().split()
            self.refs.append(line[3])
            self.alts.append(line[4].split(','))
            self.afs.append([float(x) for x in line[7].split(';')[2].split('=')[1].split(',')])
            gt = []
            gl = []
            tg = []
            mlg = []
            gl10 = [] 
            for cell in line[9: -1]: # exclude outgroup cell
                cell = cell.split(':')
                gt.append(cell[0])
                # i don't know why but sometimes there is no gl field in vcf (probably the bugs in cellcoal)
                # gl.append([float(x) if x != '-inf' else -np.inf for x in cell[5].split(',')])
                gl10.append([float(x) if x != '-inf' else -np.inf for x in cell[3].split(',')])
                tg.append(cell[-1])
                mlg.append(cell[-4]) 
            self.gts.append(gt)
            self.gls.append(gl)
            self.tgs.append(tg)
            self.mlgs.append(mlg)
            self.gl10s.append(gl10)
        vcf.close()       


    def get_PL(self):
        dic = {'AA':0, 'AC':1, 'AG':2, 'AT':3, 'CC':4, 'CG':5, 'CT':6, 'GG':7, 'GT':8, 'TT':9}
        self.pls = []
        for i in range(self.nsite):
            pll = []
            ref = self.refs[i]
            alt = self.alts[i]
            alleles = [ref] + alt
            for j in range(self.ncell):
                gl10 = self.gl10s[i][j]
                gl = []
                for p in range(len(alleles)):
                    for q in range(p+1):
                        gt = alleles[p] + alleles[q] if alleles[p] < alleles[q] else alleles[q] + alleles[p]
                        gl.append(gl10[dic[gt]])
                pl = [int(np.round(x*-10)) if x != -np.inf else 2147483647 for x in gl]
                pll.append(pl)
            self.pls.append(pll)

    def get_P_10gt(self):
        def get_allele_freq(i):
            # use ML genotype to get allele freq
            mlg = self.mlgs[i]
            count = [0, 0, 0, 0]
            for gt in mlg:
                if gt == './.':
                    continue
                gt = gt.split('/')
                for g in gt:
                    count[self.base2num[g]] += 1
            return [x/sum(count) for x in count]
        probs = []
        for i in range(self.nsite):
            prob = []
            af = get_allele_freq(i)
            ref = self.refs[i]
            alt = self.alts[i]
            alleles = [ref] + alt
            for j in range(self.ncell):
                gl = np.array(self.gls[i][j])
                original_gl = np.power(10, gl)
                weighted_gl = []
                idx = 0
                for p in range(len(alleles)):
                    for q in range(p+1):
                        if q == p:
                            weighted_gl.append(original_gl[idx] * af[self.base2num[alleles[p]]] * af[self.base2num[alleles[q]]])
                        else:
                            weighted_gl.append(original_gl[idx] * 2 * af[self.base2num[alleles[p]]] * af[self.base2num[alleles[q]]])
                        idx += 1
                weighted_gl = np.array(weighted_gl)
                if sum(weighted_gl)==0:
                    print(original_gl, af, ref, alt)
                weighted_gl = weighted_gl / sum(weighted_gl)
                prob.append(weighted_gl[0])
            probs.append(prob)
        self.probs = probs

    def get_P_10gt_2(self):
        dic = {'AA':0, 'AC':1, 'AG':2, 'AT':3, 'CC':4, 'CG':5, 'CT':6, 'GG':7, 'GT':8, 'TT':9}
        def get_allele_freq(i):
            # use ML genotype to get allele freq
            mlg = self.mlgs[i]
            num_dropout = 0
            count = [0, 0, 0, 0]
            for gt in mlg:
                if gt == './.':
                    num_dropout += 1
                    continue
                gt = gt.split('/')
                if gt[0] == gt[1]:
                    count[self.base2num[gt[0]]] += 1
                else:
                    count[self.base2num[gt[0]]] += 1 # try 0.5
                    count[self.base2num[gt[1]]] += 1
            count = np.array(count) + num_dropout/4
            return count / np.sum(count)
        probs = []
        for i in range(self.nsite):
            prob = []
            af = get_allele_freq(i)
            ref = self.refs[i]
            alt = self.alts[i]
            alleles = [ref] + alt
            for j in range(self.ncell):
                gl = np.array(self.gl10s[i][j]) # use 10 genotypes, because some of cells are problematic
                original_gl = np.power(10, gl)
                weighted_gl = []
                for p in range(len(alleles)):
                    for q in range(p+1):
                        gt = alleles[p] + alleles[q] if alleles[p] < alleles[q] else alleles[q] + alleles[p]
                        if q == p:
                            weighted_gl.append(original_gl[dic[gt]] * af[self.base2num[alleles[p]]] * af[self.base2num[alleles[q]]])
                        else:
                            weighted_gl.append(original_gl[dic[gt]] * 2 * af[self.base2num[alleles[p]]] * af[self.base2num[alleles[q]]])
                weighted_gl = np.array(weighted_gl)
                weighted_gl = weighted_gl / sum(weighted_gl)
                mlg = self.mlgs[i][j]
                if mlg == './.':
                    prob.append(0.5)
                else:
                    prob.append(weighted_gl[0])
            probs.append(prob)
        self.probs = probs


    def get_P_10gt_indv(self):
        dic = {'AA':0, 'AC':1, 'AG':2, 'AT':3, 'CC':4, 'CG':5, 'CT':6, 'GG':7, 'GT':8, 'TT':9}
        def get_genotype_freq(i):
            # use ML genotype to get allele freq
            mlg = self.mlgs[i]
            count = np.zeros(10)
            for gt in mlg:
                if gt == './.':
                    continue
                gt = gt.split('/')
                if gt[0] > gt[1]:
                    gt = gt[1] + gt[0]
                else:
                    gt = gt[0] + gt[1]
                if gt in dic:
                    count[dic[gt]] += 1
            return np.array([x/sum(count) for x in count])
        probs = []
        for i in range(self.nsite):
            prob = []
            gf = get_genotype_freq(i)
            for j in range(self.ncell):
                gl = np.array(self.gl10s[i][j])
                original_gl = np.power(10, gl)
                weighted_gl = original_gl * gf
                weighted_gl = weighted_gl/ sum(weighted_gl)
                mlg = self.mlgs[i][j]
                mlg = mlg.split('/')
                if mlg[0] > mlg[1]:
                    mlg = mlg[1] + mlg[0]
                else:
                    mlg = mlg[0] + mlg[1]
                # if mlg in dic:
                #     prob.append(weighted_gl[dic[mlg]])
                # else:
                #     prob.append(0.1)
                prob.append(max(weighted_gl))
            probs.append(prob)
        self.probs = probs


    def get_true_genotype_matrix(self):
        mat = np.zeros([self.nsite, self.ncell])
        for i in range(self.nsite):
            ref = self.refs[i]
            for j in range(self.ncell):
                tg = self.tgs[i][j]
                tg = tg.split('|')
                if tg[0] == tg[1] == ref:
                    mat[i][j] = 0
                else:
                    mat[i][j] = 1
        return mat
            
    def get_ml_genotype_matrix(self):
        mat = np.zeros([self.nsite, self.ncell])
        for i in range(self.nsite):
            ref = self.refs[i]
            for j in range(self.ncell):
                mlg = self.mlgs[i][j]
                if mlg == './.':
                    mat[i][j] = 2
                else:
                    mlg = mlg.split('/')
                    if mlg[0] == mlg[1] == ref:
                        mat[i][j] = 0
                    else:
                        mat[i][j] = 1
        return mat

    def count_errors(self):
        errors = 0
        for i in range(self.nsite):
            for j in range(self.ncell):
                mlg = self.mlgs[i][j]
                tg = self.tgs[i][j]
                if mlg == './.':
                    continue
                mlg = mlg.split('/')
                mlg = mlg[1]+mlg[0] if mlg[0] > mlg[1] else mlg[0]+mlg[1]
                tg = tg.split('|')
                tg = tg[1]+tg[0] if tg[0] > tg[1] else tg[0]+tg[1]
                if mlg != tg:
                    print(i+1, j+1, mlg, tg)
                    errors += 1
        return errors 
    

    def make_inputs(self, prefix):
        self.make_input_cellphy(prefix)
        self.make_input_scistree(prefix)

    def make_input_scistree(self, prefix):
        outname = f'{prefix}.prob'
        output_rows = []
        for i in range(self.nsite):
            pp = np.array(self.probs[i])
            c = np.sum((0.1 <= pp) & (pp <= 0.9))
            if c <= 1 * self.ncell:
                output_rows.append(i)

        with open(outname, 'w') as out:
            # out.write(f'HAPLOID {self.nsite} {self.ncell}')
            out.write(f'HAPLOID {len(output_rows)} {self.ncell}')
            for i in range(self.ncell):
                out.write(f' {i+1}')
            out.write('\n')
            # for i in range(self.nsite):
            # idx = 1
            for i in output_rows:
                out.write(f's{i+1}')
                # idx += 1
                for j in range(self.ncell):
                    prob = self.probs[i][j]
                    if prob <= 1e-10:
                        prob = 1e-10
                    if prob > 1-1e-10:
                        prob = 1-1e-10
                    out.write(f' {prob:.10f}')
                out.write('\n')

    def make_input_huntress(self, prefix):
        outname = f'{prefix}.SC'
        with open(outname, 'w') as out:
            out.write('cell\site')
            for i in range(self.nsite):
                out.write(f'\tSNP{i+1}')
            out.write('\n')
            for i in range(self.ncell):
                out.write(f'{i+1}')
                for j in range(self.nsite):
                    gt = self.gts[j][i]
                    if gt == '.|.':
                        out.write('\t3')
                    else:
                        if gt == '0|0':
                            out.write(f'\t{0}')
                        else:
                            out.write(f'\t{1}')
                    # mlg = self.mlgs[j][i]
                    # ref = self.refs[j]
                    # if mlg == './.':
                    #     out.write('\t3')
                    # else:
                    #     mlg = mlg.split('/')
                    #     if mlg[0] == mlg[1] == ref:
                    #         out.write(f'\t{0}')
                    #     else:
                    #         out.write(f'\t{1}')
                out.write('\n')

    def make_input_cellphy(self, prefix):
        outname = f'{prefix}.vcf'
        header = \
f'''##fileformat=VCFv4.3
##fileDate={date.today().isoformat()}
##source={self.vcf_file}
##reference=NONE
##contig=<ID=1>
##phasing=NO
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phread-scaled genotype likelihoods">
'''
        with open(outname, 'w') as out:
            out.write(header)
            out.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT')
            for i in range(self.ncell):
                out.write(f'\t{i+1}')
            out.write('\n')
            for i in range(self.nsite):
                out.write(f"1\t{i+1}\t.\t{self.refs[i]}\t{','.join(self.alts[i])}\t.\tPASS\t.\tGT:PL")
                for j in range(self.ncell):
                    PLSTR = ','.join([str(x) if x != -np.inf else '2147483647' for x in self.pls[i][j]])
                    out.write(f'\t{self.gts[i][j]}:{PLSTR}')
                out.write('\n')


def convert_single_file(dir, prefix):
    file = f'{dir}/vcf_dir/vcf.{prefix:04d}'
    print(file)
    fvcf = 0
    fprob = 0
    fsc = 0
    if not os.path.exists(f'{dir}/{prefix}.vcf'):
        fvcf = 1
    if not os.path.exists(f'{dir}/{prefix}.prob'):
        fprob = 1
    if not os.path.exists(f'{dir}/{prefix}.SC'):
        fsc = 1
    # if fvcf + fprob + fsc == 0:
    #     return
    print(f'start processing {dir}/{prefix}')
    vcf = SimulatedVCF(file)
    if fvcf:
        # print(f'start writing {dir}/{prefix}.vcf')
        vcf.make_input_cellphy(f'{dir}/{prefix}')
    if fprob:
        # print(f'start writing {dir}/{prefix}.prob')
        vcf.make_input_scistree(f'{dir}/{prefix}')

    if fsc:
        vcf.make_input_huntress(f'{dir}/{prefix}')

    tgmat = vcf.get_true_genotype_matrix()
    mlmat = vcf.get_ml_genotype_matrix()
    np.savetxt(f'{dir}/{prefix}.tg', tgmat, fmt='%d')
    np.savetxt(f'{dir}/{prefix}.ml', mlmat, fmt='%d')

            
if __name__ == '__main__':
    mp.set_start_method('spawn')
    from glob import glob 

    wd = '../simulation'
    # multiple cores
    with mp.Pool(32) as pool:
        for dir in os.listdir(wd):
            dir = os.path.join(wd, dir)
            if not os.path.isdir(dir) or dir.startswith('__'):
                continue
            print(dir)
            for prefix in range(1, 6):
                pool.apply_async(convert_single_file, args=(dir, prefix))
        pool.close()
        pool.join()


    # single core for huntress

    # for dir in os.listdir('./'):
    #     if os.path.isdir(dir):
    #         for prefix in range(1, 11):
    #             convert_single_file(dir, prefix)

        
    

    # single core
    # for dir in os.listdir('./'):
    #     if not os.path.isdir(dir):
    #         continue
    #     ncell = int(dir.split('_')[1])
    #     if ncell >= 1000:
    #         continue
        
    #     for prefix in range(1, 51):
    #         # pool.apply_async(convert_single_file, args=(dir, prefix))
    #         convert_single_file(dir, prefix)


    # vcf = SimulatedVCF('./ncell_1000_nsite_5000_dropout_0.2_coverage_10_doublet_0_seqerror_0.01/results/vcf_dir/vcf.0001')

        
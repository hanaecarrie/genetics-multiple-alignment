# -*- coding: utf-8 -*-

import numpy as np
import itertools

from utils import cost, addgap


def align_nw_2by2(seq1, seq2):
    """ Perform 2 by 2 sequence alignment with the Needleman and Wunsch algorithm.
    Parameters:
    ----------
    seq1: string of characters of len=n1
        sequence of nucleotides
    seq2: string of characters of len=n2
        sequence of nucleotides
    Return:
    ------
    scores: matrix of floats of dim (n1+1, n2+1)
        
    paths: matrix of floats of dim (n1+1, n2+1)
        
    aligned: list of 2 strings
        contains the aligned seq1 and seq2 respectively 
    L: list of int
        list of the positions of the inserted gaps
    """
    n1, n2 = len(seq1), len(seq2)
    # initialization: path matrix, score matrix, aligned sequences list, gaps list
    paths = np.zeros((n1 + 1, n2 + 1))
    scores = np.zeros((n1 + 1, n2 + 1))
    aligned = ["", ""]
    L = []
    for i in range(1, n1 + 1):  # browsing seq1 indexes
        scores[i, 0] = scores[i - 1, 0] - 3
        paths[i, 0] = 3
        for j in range(1, n2 + 1):  # browsing seq2 indexes
            scores[0, j] = scores[0, j - 1] - 3
            paths[0, j] = 1
            c1 = scores[i - 1, j - 1] + cost(seq1[i - 1], seq2[j - 1])
            c2 = scores[i - 1, j] - 3
            c3 = scores[i, j - 1] - 3
            scores[i, j] = max(c1, c2, c3)
            if scores[i, j] == c1:
                paths[i, j] = 2
            elif scores[i, j] == c2:
                paths[i, j] = 3
            elif scores[i, j] == c3:
                paths[i, j] = 1

    while i != 0 or j != 0:
        if paths[i, j] == 1:
            aligned[0] += '_'
            aligned[1] += seq2[j - 1]
            j = j - 1
        elif paths[i, j] == 2:
            aligned[0] += seq1[i - 1]
            aligned[1] += seq2[j - 1]
            j = j - 1
            i = i - 1
        elif paths[i, j] == 3:
            aligned[0] += seq1[i - 1]
            aligned[1] += '_'
            L.append(j)  # save gaps introduced by alignment
            i = i - 1
    aligned[0] = aligned[0][::-1]
    aligned[1] = aligned[1][::-1]
    return scores, paths, aligned, L


def align_star_multiple(seqs, verbose=2):
    """ Perform 2 by 2 sequence alignment with the Needleman and Wunsch algorithm.
    Parameters:
    ----------
    seqs: list of strings of characters
        list of sequences of nucleotides to align
    verbose: int (default=1)
        level of verbosity
    Return:
    ------
    res: 
    """
    # enumerate all possible combinations of 2 sequences
    pairs = list(itertools.combinations(list(np.arange(len(seqs))), 2))
    if verbose > 1:
        print("pairs =", pairs)
    # compute 2 by 2 scores for all combinaisons
    scores2a2 = pairs.copy()
    for ip, p in enumerate(pairs):
        scores2a2[ip] = align_nw_2by2(seqs[p[0]], seqs[p[1]])[0][-1, -1]
    if verbose > 1:
        print("scores2a2 =", scores2a2)
    # compute global scores for each sequence as sum of 2by2 scores
    scores = [0]*len(seqs)
    for i in range(len(seqs)):
        scores[i] = np.sum([sc for sci, sc in enumerate(scores2a2) if i in pairs[sci]])
    if verbose > 1:
        print("global scores =", scores)
    # choose pivot sequence as the one with the best (highest) global score
    imax = scores.index(max(scores))  # index of the pivot sequence
    pivot = seqs[imax]  # pivot sequence
    if verbose > 0:
        print("pivot sequence = ", pivot)
    # align all w.r.t pivot
    if verbose > 0:
        print("initial seqs =")
        print("\n".join(str(s) for s in seqs))
    res = seqs.copy()
    # align all sequences all together
    for ir in range(len(res)):
        # align sequences to pivot
        if res[ir] != pivot:  # the pivot sequence does not need to be aligned to itself
            alignment = align_nw_2by2(res[ir], res[imax])
            res[ir], res[imax] = alignment[2]  # result of the alignment
            L = alignment[3]  # result list gap position to consider
            if res[imax] != pivot:  # if new gaps were introduced into the pivot sequence with the 2by2 alignment
                pivot = res[imax]  # update the pivot
            # and update all the other sequences by inserting the same gaps (positions stored in L)
            for i in range(ir):
                if res[i] != pivot:
                    res[i] = addgap(res[i], L)
    if verbose > 0:
        print("res aligned to pivot =")
        print("\n".join(str(r) for r in res))
    return res


if __name__ == "__main__":
    from numpy.random import choice

    ex = "ATGAGAT"
    print(addgap(ex, [5, 1, 0, 0, 0]))

    print("EXEMPLE 0 : 2a2")
    print(align_nw_2by2("ATGAGAT", "AGGAGAGT"))
    print("EXEMPLE 1 : COURS PAGE 16")
    align_star_multiple(["ATGAGAT", "AGGAGAGT", "GGAGG", "AGGGAGT", "AGAAC"])
    print("EXEMPLE 2 : TEST SUR DONNÉES SYNTHÉTISÉES")
    alphabet = ["A", "C", "G", "T"]
    seq0 = choice(alphabet, 7)
    seq0 = ''.join(seq0)
    print(seq0)
    # seq 1 = substitution middle
    seq1 = seq0[:]
    alphabet.remove(seq1[3])
    seq1 = list(seq1)
    seq1[3] = choice(alphabet, 1)[0]
    seq1 = "".join(seq1)
    print(seq1)
    # seq 2 = deletion pos 5 and end
    seq2 = list(seq0[:])
    seq2.remove(seq2[4])
    seq2.remove(seq2[5])
    seq2 = "".join(seq2)
    print(seq2)
    # seq 3 = insertion T beginning and insertion middle
    seq3 = seq0[:]
    seq3 = "T" + seq3
    seq3 = seq3[:4] + 'G' + seq3[4:]
    print(seq3)
    # seq 4 = double insertion AC end and insertion middle
    seq4 = seq0[:]
    seq4 = seq4 + "AC"
    seq4 = seq4[:4] + 'T' + seq4[4:]
    print(seq4)
    # global alignement
    align_star_multiple([seq0, seq1, seq2, seq3, seq4])
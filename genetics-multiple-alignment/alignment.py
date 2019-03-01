# -*- coding: utf-8 -*-

import numpy as np
import itertools

from utils import cost, addgap


def align_2by2(seq1, seq2):
    n1, n2 = len(seq1), len(seq2)
    paths = np.zeros((n1 + 1, n2 + 1))
    scores = np.zeros((n1 + 1, n2 + 1))
    aligned = ["", ""]
    L = []
    for i in range(1, n1 + 1):
        scores[i, 0] = scores[i - 1, 0] - 3
        paths[i, 0] = 3
        for j in range(1, n2 + 1):
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


def star_align_multiple(seqs):
    # compute 2 by 2 scores for all possible combinations
    pairs = list(itertools.combinations(list(np.arange(len(seqs))), 2))
    print("pairs =", pairs)
    scores2a2 = pairs.copy()
    for ip, p in enumerate(pairs):
        scores2a2[ip] = align_2by2(seqs[p[0]], seqs[p[1]])[0][-1, -1]
    print("scores2a2 =", scores2a2)
    # compute global scores for every sequence
    scores = [0]*len(seqs)
    for i in range(len(seqs)):
        scores[i] = np.sum([sc for sci, sc in enumerate(scores2a2) if i in pairs[sci]])
    print("global scores =", scores)
    # pivot sequence has the best global score
    imax = scores.index(max(scores))
    pivot = seqs[imax]
    print("pivot sequence = ", pivot)
    # align all w.r.t pivot
    print("initial seqs =")
    print("\n".join(str(s) for s in seqs))
    res = seqs.copy()
    # align all together
    for ir in range(len(res)):
        if res[ir] != pivot:
            alignment = align_2by2(res[ir], res[imax])
            res[ir], res[imax] = alignment[2]
            L = alignment[3]
            if res[imax] != pivot:
                pivot = res[imax]
            for i in range(ir):
                if res[i] != pivot:
                    res[i] = addgap(res[i], L)
        print(res[ir], res[imax])
    print("res aligned to pivot =")
    print("\n".join(str(r) for r in res))
    return res


if __name__ == "__main__":
    from numpy.random import choice

    ex = "ATGAGAT"
    print(addgap(ex, [5, 1, 0, 0, 0]))

    print("EXEMPLE 0 : 2a2")
    print(align_2by2("ATGAGAT", "AGGAGAGT"))
    print("EXEMPLE 1 : COURS PAGE 16")
    star_align_multiple(["ATGAGAT", "AGGAGAGT", "GGAGG", "AGGGAGT", "AGAAC"])
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
    star_align_multiple([seq0, seq1, seq2, seq3, seq4])
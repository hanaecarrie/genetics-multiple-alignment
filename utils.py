# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from numpy.random import choice


def cost(letter1, letter2):
    """ Calculate the cost or score of the alignment of 2 letters.
    Parameters:
    ----------
    letter1: character
    letter2: character
    Return:
    ------
    score: int
        score of the alignment (-1 if the 2 letters are different, 2 if they are similar)
    """
    if letter1 == letter2:
        return 2
    else:
        return -1


def addgap(seq, gaplist, verbose=0):
    """ Insert gaps into sequence.
    Parameters:
    ----------
    seq: string of characters
        sequence of nucleotides
    gaplist: list of int
        list of gaps to insert successively in the sequence seq
    Return:
    ------
    newseq: string of characters
        the modified sequence with inserted gaps
    """
    newseq = seq[:]
    for g in gaplist:
        if g > len(newseq):
            print("gap postion bigger than sequence length -> gap inserted at the end of the sequence")
        newseq = newseq[:g] + '_' + newseq[g:]
        if verbose > 0:
            print("gap introduced in {}th position -> new sequence equals {}.".format(g, newseq))
    return newseq


def substitution(seq, pos, verbose=0, value=None, alphabet=["A", "C", "G", "T"]):
    """ Induce a mutation of the sequence by substituting a letter.
    Parameters:
    ----------
    seq: string of characters
        sequence of nucleotides
    pos: int in [-len(seq), len(seq)]
        position of the mutation
    verbose: int (default=0)
        level of verbosity
    value: None or alphabet item (default=None)
        new letter induced by the mutation.
        if None, an item different from the initial one is randomly chosen in alphabet.
    alphabet: list of characters (default=["A", "C", "G", "T"])
        list of nucleotides to consider in sequences
    Return:
    ------
    seqr: string
        the modified sequence containing the mutation
    """
    seqr = list(seq[:])
    alphabis = alphabet.copy()
    alphabis.remove(seqr[pos])
    if value == None:
        seqr[pos] = choice(alphabis, 1)[0]
    elif value in alphabet:
        seqr[pos] = value
    else:
        return "error"
    seqr = "".join(seqr)
    if verbose > 0:
        print(seqr)
    return seqr


def insertion(seq, pos, verbose=0, value=None, alphabet=["A", "C", "G", "T"]):
    """ Induce a mutation of the sequence by inserting a new letter.
    Parameters:
    ----------
    seq: string of characters
        sequence of nucleotides
    pos: int in [-len(seq), len(seq)]
        position of the mutation
    verbose: int (default=0)
        level of verbosity
    value: None or alphabet item (default=None)
        new letter induced by the mutation.
        if None, an item different from the initial one is randomly chosen in alphabet.
    alphabet: list of characters (default=["A", "C", "G", "T"])
        list of nucleotides to consider in sequences
    Return:
    ------
    seqr: string
        the modified sequence containing the mutation
    """
    seqr = seq[:]
    if value is None:
        value = choice(alphabet, 1)[0]
    seqr = seqr[:pos] + value + seqr[pos:]
    seqr = "".join(seqr)
    if verbose > 0:
        print(seqr)
    return seqr


def deletion(seq, pos, verbose=0):
    """ Induce a mutation of the sequence by deleting a letter.
    Parameters:
    ----------
    seq: string of characters
        sequence of nucleotides
    pos: int in [-len(seq), len(seq)]
        position of the mutation
    verbose: int (default=0)
        level of verbosity
    Return:
    ------
    seqr: string
        the modified sequence containing the mutation
    """
    seqr = list(seq[:])
    del seqr[pos]
    seqr = "".join(seqr)
    if verbose > 0:
        print(seqr)
    return seqr


def print_table(table):
    """ Display the score or path matrix as a table.
    Parameters:
    ----------
    table: matrix of floats or int
        score or path matrix
    Return:
    ------
    none
    """
    plt.figure()
    tb = plt.table(cellText=table.astype(int), loc=(0,0), cellLoc='center')
    tc = tb.properties()['child_artists']
    for cell in tc:
        cell.set_height(1/table.shape[0])
        cell.set_width(1/table.shape[1])
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


if __name__ == "__main__":

    alphabet = ["A", "C", "G", "T"]

    print("TEST ADDGAP FUNCTION")
    seq = choice(alphabet, 7)
    seq = ''.join(seq)
    gaplist = [1, 4, 7, 0, 0]
    newseq = addgap(seq, gaplist)
    print("The initial sequence equals: {} and the gaplist equals: {}. "
          "The final sequence equals: {}.".format(seq, gaplist, newseq))

    print("TEST COST FUNCTION")
    for i in range(5):
        l1 = choice(alphabet)
        l2 = choice(alphabet)
        print("The cost of ({}, {}) equals {}.".format(l1, l2, cost(l1, l2)))


# -*- coding: utf-8 -*-


def cost(letter1, letter2):
    if letter1 == letter2:
        return 2
    else:
        return -1


def addgap(seq, gaplist):
    for g in gaplist:
        seq = seq[:g] + '_' + seq[g:]
    return seq


if __name__ == "__main__":
    from numpy.random import choice

    alphabet = ["A", "C", "G", "T"]

    print("TEST ADDGAP FUNCTION")
    seq = choice(alphabet, 7)
    seq = ''.join(seq)
    gaplist = [1,4,7,0,0]
    newseq = addgap(seq, gaplist)
    print("The initial sequence equals: {} and the gaplist equals: {}. "
          "The final sequence equals: {}.".format(seq, gaplist, newseq))

    print("TEST COST FUNCTION")
    for i in range(5):
        l1 = choice(alphabet)
        l2 = choice(alphabet)
        print("The cost of ({}, {}) equals {}.".format(l1, l2, cost(l1, l2)))


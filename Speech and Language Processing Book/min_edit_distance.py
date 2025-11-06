# -*- coding: utf-8 -*-

def compute_ed(word1, word2):
    ed_table = [[0 for i in range(len(word2) + 1)] for j in range(len(word1) + 1)]
    ed_table[0][0] = 0
    trace_table = [[[] for i in range(len(word2) + 1)] for j in range(len(word1) + 1)]
    for i in range(len(word1) + 1):
        for j in range(len(word2) + 1):
            if i == 0:
                ed_table[0][j] = j
                if j > 0:
                    trace_table[0][j].append("Insert {}".format(word2[j - 1]))
            elif j == 0:
                ed_table[i][0] = i
                if i > 0:
                    trace_table[i][0].append("Detele {}".format(word1[i - 1]))
            else:
                ed_table[i][j] = min(ed_table[i - 1][j] + 1, ed_table[i][j - 1] + 1, \
                                     ed_table[i - 1][j - 1] + (word1[i - 1] != word2[j - 1]))
                if ed_table[i - 1][j] + 1 == ed_table[i][j]:
                    trace_table[i][j] = trace_table[i - 1][j] + ["Delete {}".format(word1[i - 1])]
                elif ed_table[i][j - 1] + 1 == ed_table[i][j]:
                    trace_table[i][j] = trace_table[i][j - 1] + ["Insert {}".format(word2[j - 1])]
                elif word1[i - 1] != word2[j - 1]:
                    trace_table[i][j] = trace_table[i - 1][j - 1] + ["Substitute {0} with {1}".format(word1[i - 1], word2[j - 1])]
                else:
                    trace_table[i][j] = trace_table[i - 1][j - 1]

    return ed_table[-1][-1], trace_table[-1][-1]


word1 = "drive"
word2 = "divers"
ed, trace = compute_ed(word1, word2)
for tr in trace:
    print(tr)

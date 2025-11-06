# Computes the minimum edit distance between two strings and the alignment

def compute_ed(word1, word2):
    m, n = len(word1), len(word2)
    ed_table = [[0] * (n + 1) for _ in range(m + 1)]
    trace_table = [[[] for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize first row (insertions)
    for j in range(1, n + 1):
        ed_table[0][j] = j
        trace_table[0][j] = trace_table[0][j - 1] + [f"Insert {word2[j - 1]}"]

    # Initialize first column (deletions)
    for i in range(1, m + 1):
        ed_table[i][0] = i
        trace_table[i][0] = trace_table[i - 1][0] + [f"Delete {word1[i - 1]}"]

    # Fill DP with pointers
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            si, tj = word1[i - 1], word2[j - 1]
            cost_sub = 0 if si == tj else 2  # substitution cost

            # Candidates: delete, insert, match/substitute
            del_cost = ed_table[i - 1][j] + 1
            ins_cost = ed_table[i][j - 1] + 1
            diag_cost = ed_table[i - 1][j - 1] + cost_sub

            best_cost = diag_cost
            op = 'M' if cost_sub == 0 else 'S'

            if del_cost < best_cost:
                best_cost, op = del_cost, 'D'
            if ins_cost < best_cost:
                best_cost, op = ins_cost, 'I'

            ed_table[i][j] = best_cost
            if op == 'D':
                trace_table[i][j] = trace_table[i - 1][j] + [f"Delete {si}"]
            elif op == 'I':
                trace_table[i][j] = trace_table[i][j - 1] + [f"Insert {tj}"]
            elif op == 'S':
                trace_table[i][j] = trace_table[i - 1][j - 1] + [f"Substitute {si} with {tj}"]
            else:  # 'M'
                trace_table[i][j] = trace_table[i - 1][j - 1]

    return ed_table[m][n], trace_table[m][n]


# Test examples
word1 = "intention"
word2 = "execution"
ed, trace = compute_ed(word1, word2)
print(f"{word1} -> {word2}")
print("distance:", ed)
for tr in trace:
    print(tr)

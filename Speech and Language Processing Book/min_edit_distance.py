# Computes the minimum edit distance between two strings and the alignment
import numpy as np


def compute_ed(word1, word2):
    m, n = len(word1), len(word2)
    # DP (int) and backpointer (single-char codes) matrices
    dp = np.zeros((m + 1, n + 1), dtype=int)
    bt = np.empty((m + 1, n + 1), dtype='U1')  # 'M','S','I','D'

    # Initialize borders
    for i in range(1, m + 1):
        dp[i, 0] = i
        bt[i, 0] = 'D'  # delete to move up
    for j in range(1, n + 1):
        dp[0, j] = j
        bt[0, j] = 'I'  # insert to move left

    # Fill DP
    for i in range(1, m + 1):
        si = word1[i - 1]
        for j in range(1, n + 1):
            tj = word2[j - 1]
            cost_sub = 0 if si == tj else 2  # substitution cost 2 in this version

            # Candidate costs
            diag = dp[i - 1, j - 1] + cost_sub
            dele = dp[i - 1, j] + 1
            ins = dp[i, j - 1] + 1

            # Choose best with tie-break preference: diag > del > ins
            best = diag
            op = 'M' if cost_sub == 0 else 'S'
            if dele < best:
                best, op = dele, 'D'
            if ins < best:
                best, op = ins, 'I'

            dp[i, j] = best
            bt[i, j] = op

    # Backtrace to build human-readable operations
    i, j = m, n
    trace = []
    while i > 0 or j > 0:
        op = bt[i, j] if (i >= 0 and j >= 0) else ''
        if i > 0 and j > 0 and op in ('M', 'S'):
            si, tj = word1[i - 1], word2[j - 1]
            if op == 'S':
                trace.append(f"Substitute {si} with {tj}")
            # for 'M' no operation text (cost 0), but keep alignment implicit
            i -= 1;
            j -= 1
        elif i > 0 and (j == 0 or op == 'D'):
            si = word1[i - 1]
            trace.append(f"Delete {si}")
            i -= 1
        else:  # Insert
            tj = word2[j - 1]
            trace.append(f"Insert {tj}")
            j -= 1

    trace.reverse()
    return int(dp[m, n]), trace


# Test examples
word1 = "intention"
word2 = "execution"
ed, trace = compute_ed(word1, word2)
print(f"{word1} -> {word2}")
print("min distance:", ed)
print("trace:")
for tr in trace:
    print(tr)

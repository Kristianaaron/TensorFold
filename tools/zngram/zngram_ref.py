# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kristian Aaron (Krishando)
class ZNgram:
    def __init__(self):
        self.reset()

    def reset(self):
        self.toks = []
        self.length = [0]
        self.link = [-1]
        self.endp = [-1]
        self.next = [{}]
        self.last = 0
        self.match_len = 0
        self.match_end = -1

    def append(self, tok):
        i = len(self.toks)
        self.toks.append(tok)
        cur = len(self.length)
        self.length.append(self.length[self.last] + 1)
        self.link.append(0)
        self.endp.append(i)
        self.next.append({})
        p = self.last
        while p != -1 and tok not in self.next[p]:
            self.next[p][tok] = cur
            p = self.link[p]
        if p == -1:
            self.link[cur] = 0
        else:
            q = self.next[p][tok]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = len(self.length)
                self.length.append(self.length[p] + 1)
                self.link.append(self.link[q])
                self.endp.append(self.endp[q])
                self.next.append(self.next[q].copy())
                while p != -1 and self.next[p].get(tok) == q:
                    self.next[p][tok] = clone
                    p = self.link[p]
                self.link[q] = self.link[cur] = clone
        self.last = cur
        v = self.link[cur]
        if v == 0:
            self.match_len, self.match_end = 0, -1
        else:
            self.match_len, self.match_end = self.length[v], self.endp[v]
            self.endp[v] = i

    def propose(self, k, min_match):
        if not k or not self.match_len or self.match_len < min_match:
            return []
        start = self.match_end + 1
        return self.toks[start:start + min(k, len(self.toks) - start)]

    def sync(self, context):
        n = len(self.toks)
        check = min(16, n)
        keep = n <= len(context) and self.toks == context[:n]
        if keep and check:
            keep = self.toks[n - check:] == context[n - check:n]
        if not keep:
            self.reset()
            n = 0
        for tok in context[n:]:
            self.append(tok)
        return len(context) - n

    def __len__(self):
        return len(self.toks)

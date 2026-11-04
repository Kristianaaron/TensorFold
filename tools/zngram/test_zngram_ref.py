# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kristian Aaron (Krishando)
import random
import unittest

from zngram_ref import ZNgram


def check_text(tokens):
    sam = ZNgram()
    for n, tok in enumerate(tokens, 1):
        sam.append(tok)
        best, ends = 0, []
        for size in range(1, n):
            suffix = tokens[n - size:n]
            found = [e for e in range(size - 1, n - 1)
                     if tokens[e - size + 1:e + 1] == suffix]
            if found:
                best, ends = size, found
        assert sam.match_len == best, (tokens[:n], sam.match_len, best)
        if best:
            e = sam.match_end
            assert e < n - 1 and tokens[e - best + 1:e + 1] == tokens[n - best:n]
            for k, m in ((0, 1), (1, 1), (4, 1), (4, best), (4, best + 1)):
                expected = tokens[e + 1:e + 1 + min(k, n - 1 - e)] if k and best >= m else []
                assert sam.propose(k, m) == expected
        else:
            assert sam.match_end == -1 and sam.propose(5, 0) == []


class ZNgramTests(unittest.TestCase):
    def test_hand_examples(self):
        s = ZNgram()
        expected = [(0, -1), (0, -1), (1, 0), (2, 1), (3, 2)]
        for tok, pair in zip([1, 2, 1, 2, 1], expected):
            s.append(tok)
            self.assertEqual((s.match_len, s.match_end), pair)
        self.assertEqual(s.propose(2, 1), [2, 1])
        s = ZNgram()
        for tok in [10, 11, 12, 13, 10, 11, 12, 14, 10, 11, 12]:
            s.append(tok)
        self.assertEqual((s.match_len, s.match_end), (3, 6))
        self.assertEqual(s.propose(4, 1), [14, 10, 11, 12])
        self.assertEqual(s.propose(4, 4), [])

    def test_repeated_and_empty(self):
        s = ZNgram()
        self.assertEqual(s.propose(3, 1), [])
        for k in range(1, 51):
            s.append(9)
            self.assertEqual((s.match_len, s.match_end), (k - 1, k - 2))
            self.assertLessEqual(len(s.length), 2 * k)

    def test_oracle_and_random(self):
        for seed in range(300):
            rng = random.Random(seed)
            alphabet = [2, 3, 5, 300][seed % 4]
            check_text([rng.randrange(alphabet) for _ in range(rng.randint(1, 150))])
        check_text([0xFFFFFFFF, 0, 0xFFFFFFFF, 0])

    def test_sync(self):
        context = list(range(60))
        s = ZNgram()
        self.assertEqual(s.sync(context[:40]), 40)
        self.assertEqual(s.sync(context[:55]), 15)
        changed = context[:55]
        changed[10] = 999
        self.assertEqual(s.sync(changed), len(changed))
        self.assertEqual(s.sync(changed[:20]), 20)
        self.assertEqual(s.toks, changed[:20])
        tail_match = context[:55]
        s.reset()
        s.sync(tail_match)
        tail_match[10] = 999
        self.assertEqual(s.sync(tail_match), len(tail_match))
        self.assertEqual(s.toks, tail_match)


if __name__ == "__main__":
    unittest.main()

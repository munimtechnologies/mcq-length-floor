"""Unit tests for mcq-length-floor (hand-traced; standard library only). Apache-2.0."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from fractions import Fraction

import mcq_length_floor as m

F = Fraction


class TestCredits(unittest.TestCase):
    def test_chars(self):
        self.assertEqual(m.credits(["aa", "b", "cccc"], 2), (F(1), F(0)))
        self.assertEqual(m.credits(["xx", "yy", "z", "w"], 0), (F(1, 2), F(0)))
        self.assertEqual(m.credits(["p", "qqq", "rr"], 0), (F(0), F(1)))
        self.assertEqual(m.credits([" a ", "b", "cc"], 1), (F(0), F(1, 2)))

    def test_words(self):
        self.assertEqual(m.credits(["one two three", "four", "five six"], 1, "words"), (F(0), F(1)))

    def test_polarity(self):
        self.assertEqual(m.polarity("The patient does not smoke. What is the diagnosis?"), "pos")
        self.assertEqual(m.polarity("All of the following are true EXCEPT:"), "neg")
        self.assertEqual(m.polarity("Which isn't seen?"), "neg")
        self.assertEqual(m.polarity("Nothing is known about the knot."), "pos")
        self.assertEqual(m.polarity(None), "pos")


class TestExtract(unittest.TestCase):
    def ex(self, rec, **kw):
        kw.setdefault("options_col", "options")
        return m.extract(rec, **kw)

    def test_shapes(self):
        self.assertEqual(self.ex({"options": ["a", "bb", "N/A"], "answer": 1}, pad_tokens=("N/A",)),
                         (["a", "bb"], 1, None))
        self.assertEqual(self.ex({"options": ["a", "bb", "ccc"], "answer": "c"}), (["a", "bb", "ccc"], 2, None))
        self.assertEqual(self.ex({"options": ["x", "yy"], "answer": "yy"}), (["x", "yy"], 1, None))
        self.assertEqual(self.ex({"options": ["x", "yy"], "answer": "1"}, one_based=True), (["x", "yy"], 0, None))
        self.assertEqual(self.ex({"options": {"B": "bb", "A": "a"}, "answer": "B"}), (["a", "bb"], 1, None))
        self.assertEqual(self.ex({"options": '["a", "bb"]', "answer": 0}), (["a", "bb"], 0, None))
        self.assertEqual(self.ex({"options": ["a", "", "ccc"], "answer": 2}), (["a", "ccc"], 1, None))
        self.assertEqual(m.extract({"opa": "p", "opb": "qq", "cop": 1}, option_cols=["opa", "opb"], answer_col="cop"),
                         (["p", "qq"], 1, None))

    def test_rejects(self):
        self.assertEqual(self.ex({"options": "notjson", "answer": 0})[2], "options_not_a_list")
        self.assertEqual(self.ex({"options": ["a", "b"], "answer": 5})[2], "bad_answer")
        self.assertEqual(self.ex({"options": ["a", "b"], "answer": True})[2], "bad_answer")
        self.assertEqual(self.ex({"options": ["a", "N/A"], "answer": 1}, pad_tokens=("N/A",))[2], "gold_blank")
        self.assertEqual(self.ex({"options": ["a", "", ""], "answer": 0})[2], "fewer_than_2_options")


ROWS = [
    {"options": ["aa", "b", "cccc"], "answer": 2, "cat": "x", "q": "Which is true?"},
    {"options": ["xx", "yy", "z", "w"], "answer": 0, "cat": "x", "q": "All EXCEPT:"},
    {"options": ["p", "qqq", "rr"], "answer": 0, "cat": "y", "q": "Which is not seen?"},
    {"options": ["a"], "answer": 0, "cat": "y"},
]


class TestCensus(unittest.TestCase):
    def setUp(self):
        self.rep, self.skipped = m.census(ROWS, options_col="options", answer_col="answer", groups=["cat"],
                                          question_col="q", split_polarity=True)

    def test_order_and_skips(self):
        self.assertEqual(list(self.rep), ["ALL", "ALL | stem=neg", "ALL | stem=pos", "cat=x", "cat=x | stem=neg",
                                          "cat=x | stem=pos", "cat=y", "cat=y | stem=neg"])
        self.assertEqual(self.skipped, {"fewer_than_2_options": 1})

    def test_all(self):
        r = self.rep["ALL"]
        self.assertEqual(r["n"], 3)
        self.assertAlmostEqual(r["pick_longest"], 0.5, places=12)
        self.assertAlmostEqual(r["pick_shortest"], 1 / 3, places=12)
        self.assertAlmostEqual(r["chance"], 11 / 36, places=12)
        self.assertAlmostEqual(r["lift"], 7 / 36, places=12)
        self.assertAlmostEqual(r["ratio"], 18 / 11, places=12)
        self.assertAlmostEqual(r["se"], 109 ** 0.5 / 36, places=12)
        self.assertAlmostEqual(r["z"], 7 / 109 ** 0.5, places=12)

    def test_slices(self):
        x = self.rep["cat=x"]
        self.assertAlmostEqual(x["pick_longest"], 0.75, places=12)
        self.assertAlmostEqual(x["lift"], 11 / 24, places=12)
        self.assertAlmostEqual(x["ratio"], 18 / 7, places=12)
        y = self.rep["cat=y"]
        self.assertEqual((y["n"], y["se"], y["z"]), (1, None, None))
        self.assertAlmostEqual(y["lift"], -1 / 3, places=12)
        neg = self.rep["ALL | stem=neg"]
        self.assertEqual(neg["n"], 2)
        self.assertAlmostEqual(neg["pick_longest"], 0.25, places=12)
        self.assertAlmostEqual(neg["pick_shortest"], 0.5, places=12)
        self.assertAlmostEqual(neg["lift"], -1 / 24, places=12)
        self.assertEqual(self.rep["ALL | stem=pos"]["pick_longest"], 1.0)

    def test_empty_report(self):
        self.assertEqual(m.report(m.new_acc()), {"n": 0})


class TestCli(unittest.TestCase):
    def run_main(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(m.main(argv), 0)
        return buf.getvalue()

    def test_csv_markdown(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.csv")
            with open(p, "w", encoding="utf-8", newline="") as fh:
                fh.write("opa,opb,cop\np,qq,1\nrrr,s,0\n")
            out = self.run_main(["--csv", p, "--option-cols", "opa,opb", "--answer", "cop"])
        self.assertIn("| ALL | 2 | 0.5000 | 1.0000 | 0.0000 | +0.5000 | 2.00 | - |", out)

    def test_jsonl_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.jsonl")
            with open(p, "w", encoding="utf-8") as fh:
                for r in ROWS:
                    fh.write(json.dumps(r) + "\n")
            out = self.run_main(["--jsonl", p, "--options", "options", "--answer", "answer", "--group", "cat",
                                 "--json"])
        data = json.loads(out)
        self.assertEqual(data["reports"]["ALL"]["n"], 3)
        self.assertEqual(data["skipped"], {"fewer_than_2_options": 1})
        self.assertEqual(sorted(data["reports"]), ["ALL", "cat=x", "cat=y"])


if __name__ == "__main__":
    unittest.main()

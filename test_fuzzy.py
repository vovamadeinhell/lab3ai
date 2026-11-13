"""Проверки формул, независимых численных примеров и полноты баз знаний."""
from dataclasses import replace
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from fuzzy import (IMPLICATIONS, Membership, Rule, System, Variable, aggregate,
                   centroid, infer, load_system)


BASE = Path(__file__).resolve().parent


class FuzzyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single = load_system(BASE / "single_input.txt")
        cls.multi = load_system(BASE / "multi_input.txt")

    def test_membership_and_shoulders(self):
        triangle = Membership("triangle", (0, 5, 10))
        self.assertEqual([triangle(x) for x in (-1, 0, 2.5, 5, 7.5, 10, 11)],
                         [0, 0, .5, 1, .5, 0, 0])
        left = Membership("triangle", (0, 0, 10))
        right = Membership("trapezoid", (0, 4, 10, 10))
        self.assertEqual(left(0), 1)
        self.assertEqual(right(10), 1)
        self.assertEqual(right(2), .5)

    def test_implication_formulas(self):
        expected = {"mamdani": .3, "larsen": .18,
                    "kleene_dienes": .4, "lukasiewicz": .7}
        for name, value in expected.items():
            self.assertAlmostEqual(IMPLICATIONS[name](.6, .3), value)

    def test_aggregation_formulas(self):
        expected = {"max": .6, "min": .3, "probor": .72, "bounded_sum": .9}
        for name, value in expected.items():
            self.assertAlmostEqual(aggregate([.3, .6], name), value)

    def test_centroid_nonuniform_grid(self):
        # Независимый аналитический ответ: центр треугольника (0+2+5)/3.
        value, area = centroid([0, 2, 5], [0, 1, 0])
        self.assertAlmostEqual(value, 7 / 3)
        self.assertAlmostEqual(area, 2.5)
        self.assertEqual(centroid([0, 1], [0, 0]), (None, 0))

    def test_seven_rule_crisp_example(self):
        result = infer(self.single, {"temperature": 77})
        self.assertEqual(result.alphas, [0, 0, 0, .2, .8, 0, 0])
        # Ручное интегрирование max(min(.2,B4),min(.8,B5)).
        self.assertAlmostEqual(result.area, 17.4)
        self.assertAlmostEqual(result.crisp, 66.37931034482759)

    def test_crisp_mechanisms_agree_off_grid(self):
        results = [infer(self.single, {"temperature": 77.123}, method=m)
                   for m in ("max_min", "max_product", "activation")]
        self.assertEqual(results[0].membership, results[1].membership)
        self.assertEqual(results[0].membership, results[2].membership)

    def test_two_compositions_independent_example(self):
        # При x=0,1,2: A'=(0,.5,1), A=(1,.5,0); B(1)=1.
        # sup min(A',A)=.5, sup A'*A=.25. Второй B(y)=y/2.
        var = Variable("x", "x", "", (0, 2), 1,
                       {"low": Membership("triangle", (0, 0, 2))})
        out = Variable("y", "y", "", (0, 2), 1,
                       {"high": Membership("triangle", (0, 2, 2))})
        sys = System("test", {"x": var}, out,
                     (Rule((("x", "low"),), "high"),), {"x": 2})
        obs = {"x": Membership("triangle", (0, 2, 2))}
        mm = infer(sys, {"x": 2}, method="max_min", fuzzy_inputs=obs)
        mp = infer(sys, {"x": 2}, method="max_product", fuzzy_inputs=obs)
        self.assertEqual(mm.membership, [0, .5, .5])
        self.assertEqual(mp.membership, [0, .25, .25])

    def test_fuzzy_mamdani_reduction(self):
        obs = {"temperature": Membership("triangle", (71, 77, 83))}
        composition = infer(self.single, self.single.example, method="max_min", fuzzy_inputs=obs)
        activation = infer(self.single, self.single.example, fuzzy_inputs=obs)
        self.assertEqual(composition.membership, activation.membership)

    def test_multi_rule_strengths(self):
        result = infer(self.multi, self.multi.example)
        active = {i: a for i, a in enumerate(result.alphas, 1) if a > 0}
        self.assertEqual(set(active), {14, 15, 17, 18, 23, 24, 26, 27})
        self.assertAlmostEqual(active[14], .64)
        self.assertAlmostEqual(active[15], .30)
        self.assertAlmostEqual(active[17], .36)
        self.assertAlmostEqual(active[27], 12 / 45)
        self.assertAlmostEqual(result.crisp, 66.84084946591212, places=4)

    def test_all_27_combinations_are_covered(self):
        self.assertEqual(len(self.single.rules), 7)
        self.assertEqual(len(self.multi.rules), 27)
        signatures = {tuple(t for _, t in rule.conditions) for rule in self.multi.rules}
        self.assertEqual(len(signatures), 27)
        for coords in itertools.product((20, 65, 110), (0, 50, 100), (0, 5, 10)):
            result = infer(self.multi, dict(zip(self.multi.inputs, coords)))
            self.assertGreater(result.area, 0)
            self.assertTrue(10 <= result.crisp <= 100)

    def test_logical_implications_do_not_skip_zero_rules(self):
        for name in ("kleene_dienes", "lukasiewicz"):
            result = infer(self.single, self.single.example, implication=name)
            self.assertTrue(all(mu == 1 for mu in result.membership))
            self.assertAlmostEqual(result.crisp, 55)
            self.assertTrue(result.warnings)

    def test_empty_output_is_reported(self):
        result = infer(self.single, self.single.example, aggregation="min")
        self.assertIsNone(result.crisp)
        self.assertEqual(result.area, 0)
        self.assertTrue(result.warnings)

    def test_grid_refinement(self):
        # Уменьшение шага в 4 раза: изменение ответа меньше 0.001 п.п.
        var = replace(self.single.inputs["temperature"], step=.0625)
        fine = replace(self.single, inputs={"temperature": var},
                       output=replace(self.single.output, step=.0625))
        obs = {"temperature": Membership("triangle", (71, 77, 83))}
        for method in ("max_min", "max_product", "activation"):
            a = infer(self.single, self.single.example, method=method, fuzzy_inputs=obs)
            b = infer(fine, fine.example, method=method, fuzzy_inputs=obs)
            self.assertLess(abs(a.crisp - b.crisp), .001)

    def test_bad_input_rejected(self):
        for values in ({}, {"temperature": 111}, {"temperature": float("nan")},
                       {"temperature": True}, {"temperature": 77, "other": 1}):
            with self.assertRaises(ValueError):
                infer(self.single, values)
        with self.assertRaises(ValueError):
            infer(self.multi, self.multi.example, method="max_min")

    def test_bad_base_rejected(self):
        data = json.loads((BASE / "single_input.txt").read_text(encoding="utf-8"))
        data["rules"][0]["then"] = "нет_такого_терма"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.txt"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_system(path)


if __name__ == "__main__":
    unittest.main()

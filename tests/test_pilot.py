import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_pilot import action_direction, collapse_directions, is_crohn, match_intervention, therapeutic_direction


class ScientificRulesTests(unittest.TestCase):
    def test_direction_inverts_risk_and_preserves_protection(self):
        expected = {("LoF", "risk"): "activate", ("LoF", "protective"): "inhibit",
                    ("GoF", "risk"): "inhibit", ("GoF", "protective"): "activate"}
        for pair, direction in expected.items():
            with self.subTest(pair=pair):
                self.assertEqual(therapeutic_direction(*pair), direction)

    def test_missing_direction_is_not_imputed(self):
        for pair in [(None, "risk"), ("LoF", None), (None, None)]:
            self.assertEqual(therapeutic_direction(*pair), "unknown")
        self.assertEqual(action_direction("MODULATOR"), "unknown")
        self.assertEqual(action_direction("BINDING AGENT"), "unknown")

    def test_conflict_is_not_resolved_by_counting_repeated_evidence(self):
        self.assertEqual(collapse_directions(["activate"] * 50 + ["inhibit"]), "conflicting")
        self.assertEqual(collapse_directions(["activate", "unknown"]), "activate")

    def test_matching_excludes_placebo_ambiguities_and_combination_guessing(self):
        aliases = {"adalimumab": {"A"}, "brand": {"A", "B"}, "methotrexate": {"M"}}
        self.assertEqual(match_intervention("Adalimumab placebo", [], aliases)[1], "placebo")
        self.assertEqual(match_intervention("Brand", [], aliases)[1], "ambiguous")
        self.assertEqual(match_intervention("Adalimumab and methotrexate", [], aliases)[1], "unmatched")
        self.assertEqual(match_intervention("Adalimumab 40 mg", [], aliases), ({"A"}, "dose_route_cleaned"))

    def test_primary_name_takes_precedence_over_broad_other_names(self):
        aliases = {"adalimumab": {"A"}, "antitnf": {"A", "B"}}
        self.assertEqual(match_intervention("Adalimumab", ["Anti-TNF"], aliases), ({"A"}, "exact"))

    def test_preferred_name_resolves_synonym_collision(self):
        aliases = {"drug": {"A", "B"}}
        self.assertEqual(match_intervention("Drug", [], aliases, {"drug": {"A"}}), ({"A"}, "exact"))

    def test_ibd_and_uc_are_not_crohn_labels(self):
        self.assertFalse(is_crohn(["Inflammatory Bowel Disease", "Ulcerative Colitis"]))
        self.assertTrue(is_crohn(["Crohn's Disease"]))
        self.assertTrue(is_crohn(["Regional Enteritis"]))


if __name__ == "__main__":
    unittest.main()

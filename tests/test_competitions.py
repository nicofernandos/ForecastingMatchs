import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def configured_list(filename, name):
    tree = ast.parse((ROOT / filename).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing {name} in {filename}")


class CompetitionTests(unittest.TestCase):
    def test_ligue1_enabled_in_prediction_and_data_download(self):
        expected = ["PL", "BL1", "SA", "PPL", "FL1"]
        self.assertTrue(set(expected).issubset(configured_list("predict.py", "COMPS")))
        self.assertTrue(set(expected).issubset(configured_list("fetch_matches.py", "COMPETITIONS")))

    def test_la_liga_enabled_in_prediction_and_data_download(self):
        self.assertIn("PD", configured_list("predict.py", "COMPS"))
        self.assertIn("PD", configured_list("fetch_matches.py", "COMPETITIONS"))

    def test_championship_enabled_in_prediction_and_data_download(self):
        self.assertIn("ELC", configured_list("predict.py", "COMPS"))
        self.assertIn("ELC", configured_list("fetch_matches.py", "COMPETITIONS"))

    def test_champions_league_enabled_in_prediction_and_data_download(self):
        self.assertIn("CL", configured_list("predict.py", "COMPS"))
        self.assertIn("CL", configured_list("fetch_matches.py", "COMPETITIONS"))


if __name__ == "__main__":
    unittest.main()

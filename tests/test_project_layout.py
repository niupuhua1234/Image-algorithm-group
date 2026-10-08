"""Fast checks for the reorganized project, without starting GPU training."""

import ast
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from competition.workflow import commands


class ProjectLayoutTests(unittest.TestCase):
    def test_entrypoints_exist(self):
        for name, command in commands().items():
            with self.subTest(action=name):
                self.assertTrue((ROOT / command[0]).is_file())

    def test_basic_is_scratch(self):
        self.assertIn("--scratch", commands()["basic-train"])
        self.assertNotIn("--initial-weight", commands()["basic-train"])

    def test_standard_transfer(self):
        train, test = commands()["transfer-train"], commands()["transfer-test"]
        self.assertEqual(train[train.index("--method") + 1], "standard")
        self.assertEqual(test[test.index("--methods") + 1], "standard")

    def test_primary_syntax(self):
        paths = {ROOT / command[0] for command in commands().values()}
        paths.update((ROOT / "competition").glob("*.py"))
        for path in paths:
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    def test_active_data(self):
        for profile in ("base", "rtdetr_gpt_low_contrast_combined_v1"):
            folder = ROOT / "datasets/prepared" / profile
            for split in ("train", "val", "test"):
                with self.subTest(profile=profile, split=split):
                    data = json.loads((folder / "annotations" / f"instances_{split}.json").read_text())
                    self.assertTrue(data["images"])
                    for item in data["images"]:
                        name = Path(item["file_name"])
                        self.assertTrue((folder / "images" / split / name).is_file())
                        self.assertTrue((folder / "voc_xml" / split / name.with_suffix(".xml")).is_file())

    def test_incremental_uses_transfer(self):
        spec = importlib.util.spec_from_file_location("incremental_entry", ROOT / "incremental_after_transfer/train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIn("standard_010shot", str(module.SOURCE))
        self.assertTrue(module.SOURCE.is_file())
        # The trainer now lives in the shared package, not the obsolete experiment directory.
        self.assertIn("from competition import incremental_trainer", (ROOT / "incremental_after_transfer/train.py").read_text())


if __name__ == "__main__":
    unittest.main()

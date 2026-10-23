"""The dependency map in docs/dependency-map stays the map of the code it is committed with."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from mimir.tests.test_architecture import RANKS

GENERATOR = Path(__file__).resolve().parents[2] / "docs" / "dependency-map" / "generate.py"


def _generator():
    spec = importlib.util.spec_from_file_location("dependency_map_generate", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DependencyMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gen = _generator()
        cls.data = cls.gen.extract()

    def test_the_page_matches_the_code(self) -> None:
        """Units, files and imports; sizes and line numbers move with every edit and are not compared."""
        on_page = self.gen.page_data(self.gen.PAGE.read_text())
        self.assertEqual(self.gen.structure(on_page), self.gen.structure(self.data),
                         "docs/dependency-map/index.html is out of date; "
                         "run: python docs/dependency-map/generate.py")

    def test_the_image_matches_the_code(self) -> None:
        self.assertEqual(self.gen.IMAGE.read_text(), self.gen.draw(self.data),
                         "docs/dependency-map/map.svg is out of date; "
                         "run: python docs/dependency-map/generate.py")

    def test_every_client_unit_is_drawn_at_its_rank(self) -> None:
        """The map reads its rows from docs/ARCHITECTURE.md, which must agree with the enforced ranks."""
        self.assertEqual({u: rank for u, (rank, _what) in self.gen.architecture_ranks().items()},
                         RANKS)
        unranked = [u["id"] for u in self.data["units"]
                    if u["area"] == "client" and u["rank"] is None]
        self.assertEqual(unranked, [])

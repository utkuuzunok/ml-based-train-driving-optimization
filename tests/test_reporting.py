import ast
import inspect
import tempfile
import unittest
from pathlib import Path

import reporting
from reporting import generate_report_figures


class ReportingTests(unittest.TestCase):
    def test_module_does_not_depend_on_simulator(self) -> None:
        syntax_tree = ast.parse(inspect.getsource(reporting))
        imported_modules = set()
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        self.assertNotIn("train_simulator", imported_modules)
        self.assertNotIn("experiments", imported_modules)

    def test_generates_four_nonempty_png_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = generate_report_figures(
                Path("results/ml"), Path(temporary_directory)
            )

            self.assertEqual(len(paths), 4)
            for path in paths.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 10_000)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()

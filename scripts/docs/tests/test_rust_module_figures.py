"""Independent arithmetic and CLI regression tests for the documentation figures."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

SCRIPT = Path(__file__).resolve().parents[1] / 'generate_rust_module_figures.py'
spec = importlib.util.spec_from_file_location('figures', SCRIPT)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


class ReferenceTests(unittest.TestCase):
    def test_ticket_cost_changes_optimum_not_inventory_constraints(self):
        # These are the page's pre-existing assertions and hand-summed scores.
        for cost, path, score in [
            (0, (0, 3, 3, 0, 3, 0), .129),
            (.01, (0, 3, 3, 0, 3, 0), .089),
            (.03, (0, 3, 3, 3, 3, 0), .042),
            (.1, (0, 0, 0, 0, 0, 0), 0),
        ]:
            actual, value, count = figures.inventory_optimum(cost)
            self.assertEqual(actual, path)
            self.assertAlmostEqual(value, score, places=12)
            self.assertEqual(count, 256)

    def test_backward_roll_preserves_latest_close(self):
        raw, adjusted, gap = figures.roll_example()
        self.assertEqual(raw, [70.4, 71.2, 71.1, 72.9, 73.0])
        for actual, expected in zip(adjusted, [71.3, 72.1, 72.0, 72.9, 73.0]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(gap, .9)
        self.assertAlmostEqual(adjusted[3] - adjusted[2], .9)
        self.assertEqual(adjusted[-1], raw[-1])

    def test_partition_work_is_discrete_not_continuous_approximation(self):
        linear, nested = figures.partition_example()
        self.assertEqual(linear, ([0, 250, 500, 750, 1000], [31125, 93625, 156125, 218625]))
        self.assertEqual(nested, ([0, 500, 707, 866, 1000], [124750, 124821, 124974, 124955]))
        self.assertEqual(sum(linear[1]), 499500)
        self.assertEqual(sum(nested[1]), 499500)

    def test_analytic_fingerprint_is_not_sample_estimate(self):
        self.assertEqual(figures.ideal_effects(-1, -1), (-2, 1, 1))
        self.assertEqual(figures.ideal_effects(0, 1), (0, 0, 0))
        self.assertEqual(figures.ideal_effects(.5, -1), (1, .25, -.5))


class ArtifactTests(unittest.TestCase):
    def cli(self, directory, mode):
        return subprocess.run([sys.executable, str(SCRIPT), mode, '--output-dir', str(directory)],
                              capture_output=True, text=True)

    def test_check_detects_missing_and_changed_artifacts_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertNotEqual(self.cli(root, '--check').returncode, 0)
            self.assertEqual(list(root.iterdir()), [])
            self.assertEqual(self.cli(root, '--write').returncode, 0)
            before = {p.name: p.read_bytes() for p in root.glob('*.svg')}
            self.assertEqual(len(before), 4)
            self.assertEqual(self.cli(root, '--check').returncode, 0)
            self.assertEqual(self.cli(root, '--write').returncode, 0)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.glob('*.svg')})
            damaged = next(root.glob('*.svg'))
            damaged.write_text('drift', encoding='utf-8')
            self.assertNotEqual(self.cli(root, '--check').returncode, 0)
            self.assertEqual(damaged.read_text(), 'drift')

    def test_svg_has_accessible_names_and_no_external_assets(self):
        ns = {'s': 'http://www.w3.org/2000/svg'}
        for name, content in figures.render_all().items():
            with self.subTest(name=name):
                root = ET.fromstring(content)
                self.assertEqual(root.attrib['role'], 'img')
                self.assertTrue(root.find('s:title', ns).text)
                self.assertTrue(root.find('s:desc', ns).text)
                self.assertEqual(root.attrib['aria-labelledby'], 'title desc')
                self.assertFalse(root.findall('.//s:image', ns))
                self.assertFalse(root.findall('.//s:script', ns))
                for node in root.iter():
                    self.assertFalse(any('href' in key for key in node.attrib))


if __name__ == '__main__':
    unittest.main()

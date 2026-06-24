import subprocess
import sys
import unittest
from pathlib import Path


class InferenceScriptExecutionTest(unittest.TestCase):
    def test_direct_script_execution_resolves_project_imports(self):
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/inference.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input_path", result.stdout)


if __name__ == "__main__":
    unittest.main()

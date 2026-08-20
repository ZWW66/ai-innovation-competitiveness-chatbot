import unittest
from pathlib import Path

import tomllib


class StreamlitConfigurationTests(unittest.TestCase):
    def test_transformers_modules_are_not_scanned_by_file_watcher(self):
        config_path = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["server"]["fileWatcherType"], "none")


if __name__ == "__main__":
    unittest.main()

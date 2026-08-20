import importlib
import sys
import unittest
from types import ModuleType
from unittest.mock import patch


class OpenAIModelConfigurationTests(unittest.TestCase):
    def test_llm_uses_token_efficient_gpt_5_6_configuration(self):
        captured = {}
        fake_crewai = ModuleType("crewai")

        def fake_llm(**kwargs):
            captured.update(kwargs)
            return kwargs

        fake_crewai.LLM = fake_llm
        sys.modules.pop("crew.llm", None)
        with patch.dict(sys.modules, {"crewai": fake_crewai}):
            module = importlib.import_module("crew.llm")

        self.assertEqual(captured["model"], "openai/gpt-5.6-luna")
        self.assertEqual(captured["reasoning_effort"], "none")
        self.assertEqual(captured["max_completion_tokens"], 1500)
        self.assertNotIn("additional_params", captured)
        self.assertEqual(module.chatgpt_llm, captured)


if __name__ == "__main__":
    unittest.main()

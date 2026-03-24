"""Unit tests for llm_provider module."""

import argparse
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_provider import (
    PROVIDER_PRESETS,
    add_provider_args,
    clamp_temperature,
    create_client,
    get_model_name,
)


class TestProviderPresets(unittest.TestCase):
    """Test provider preset configuration."""

    def test_openai_preset_exists(self):
        self.assertIn("openai", PROVIDER_PRESETS)

    def test_minimax_preset_exists(self):
        self.assertIn("minimax", PROVIDER_PRESETS)

    def test_openai_preset_values(self):
        preset = PROVIDER_PRESETS["openai"]
        self.assertIsNone(preset["base_url"])
        self.assertEqual(preset["default_model"], "gpt-4-vision-preview")
        self.assertEqual(preset["env_key"], "OPENAI_API_KEY")

    def test_minimax_preset_values(self):
        preset = PROVIDER_PRESETS["minimax"]
        self.assertEqual(preset["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(preset["default_model"], "MiniMax-M2.7")
        self.assertEqual(preset["env_key"], "MINIMAX_API_KEY")


class TestAddProviderArgs(unittest.TestCase):
    """Test add_provider_args adds correct arguments."""

    def test_adds_provider_arg(self):
        parser = argparse.ArgumentParser()
        add_provider_args(parser)
        args = parser.parse_args(["--provider", "minimax"])
        self.assertEqual(args.provider, "minimax")

    def test_default_provider_is_openai(self):
        parser = argparse.ArgumentParser()
        add_provider_args(parser)
        args = parser.parse_args([])
        self.assertEqual(args.provider, "openai")

    def test_adds_base_url_arg(self):
        parser = argparse.ArgumentParser()
        add_provider_args(parser)
        args = parser.parse_args(["--base_url", "https://custom.api.com/v1"])
        self.assertEqual(args.base_url, "https://custom.api.com/v1")

    def test_adds_model_arg(self):
        parser = argparse.ArgumentParser()
        add_provider_args(parser)
        args = parser.parse_args(["--model", "gpt-4o"])
        self.assertEqual(args.model, "gpt-4o")

    def test_invalid_provider_raises(self):
        parser = argparse.ArgumentParser()
        add_provider_args(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--provider", "invalid"])


class TestGetModelName(unittest.TestCase):
    """Test get_model_name resolution."""

    def _make_args(self, provider="openai", model=None):
        args = argparse.Namespace(provider=provider, model=model)
        return args

    def test_openai_default_model(self):
        args = self._make_args(provider="openai")
        self.assertEqual(get_model_name(args), "gpt-4-vision-preview")

    def test_minimax_default_model(self):
        args = self._make_args(provider="minimax")
        self.assertEqual(get_model_name(args), "MiniMax-M2.7")

    def test_explicit_model_overrides_default(self):
        args = self._make_args(provider="openai", model="gpt-4o")
        self.assertEqual(get_model_name(args), "gpt-4o")

    def test_explicit_model_overrides_minimax_default(self):
        args = self._make_args(provider="minimax", model="MiniMax-M2.5")
        self.assertEqual(get_model_name(args), "MiniMax-M2.5")


class TestCreateClient(unittest.TestCase):
    """Test create_client creates OpenAI client with correct params."""

    @patch("llm_provider.OpenAI")
    def test_openai_client_default(self, mock_openai_cls):
        args = argparse.Namespace(
            provider="openai", base_url=None, api_key="test-key"
        )
        create_client(args)
        mock_openai_cls.assert_called_once_with(api_key="test-key")

    @patch("llm_provider.OpenAI")
    def test_minimax_client_sets_base_url(self, mock_openai_cls):
        args = argparse.Namespace(
            provider="minimax", base_url=None, api_key="mm-key"
        )
        create_client(args)
        mock_openai_cls.assert_called_once_with(
            api_key="mm-key", base_url="https://api.minimax.io/v1"
        )

    @patch("llm_provider.OpenAI")
    def test_custom_base_url_overrides_preset(self, mock_openai_cls):
        args = argparse.Namespace(
            provider="minimax", base_url="https://custom.api.com/v1", api_key="key"
        )
        create_client(args)
        mock_openai_cls.assert_called_once_with(
            api_key="key", base_url="https://custom.api.com/v1"
        )

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "env-mm-key"}, clear=False)
    @patch("llm_provider.OpenAI")
    def test_minimax_env_key_auto_detected(self, mock_openai_cls):
        args = argparse.Namespace(
            provider="minimax", base_url=None, api_key=None
        )
        create_client(args)
        mock_openai_cls.assert_called_once_with(
            api_key="env-mm-key", base_url="https://api.minimax.io/v1"
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-oai-key"}, clear=False)
    @patch("llm_provider.OpenAI")
    def test_openai_env_key_fallback(self, mock_openai_cls):
        args = argparse.Namespace(
            provider="openai", base_url=None, api_key=None
        )
        create_client(args)
        mock_openai_cls.assert_called_once_with(api_key="env-oai-key")


class TestClampTemperature(unittest.TestCase):
    """Test temperature clamping for different providers."""

    def test_openai_no_clamping(self):
        self.assertEqual(clamp_temperature(0.0, "openai"), 0.0)
        self.assertEqual(clamp_temperature(2.0, "openai"), 2.0)

    def test_minimax_clamp_low(self):
        self.assertEqual(clamp_temperature(0.0, "minimax"), 0.01)

    def test_minimax_clamp_high(self):
        self.assertEqual(clamp_temperature(2.0, "minimax"), 1.0)

    def test_minimax_normal_range(self):
        self.assertEqual(clamp_temperature(0.5, "minimax"), 0.5)

    def test_minimax_boundary_one(self):
        self.assertEqual(clamp_temperature(1.0, "minimax"), 1.0)


class TestCallGptFunction(unittest.TestCase):
    """Test call_gpt function signature compatibility."""

    def test_call_gpt_with_client_and_model(self):
        """Verify call_gpt accepts client and model_name parameters."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test caption"
        mock_client.chat.completions.create.return_value = mock_response

        from step3_1_GPT4V_video_caption_concise import call_gpt

        result = call_gpt(
            [{"type": "text", "text": "test prompt"}],
            mock_client,
            model_name="MiniMax-M2.7",
        )
        self.assertEqual(result, "test caption")
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs["model"], "MiniMax-M2.7")


class TestSaveOutputFunction(unittest.TestCase):
    """Test save_output function signature compatibility."""

    def test_save_output_accepts_client_and_model(self):
        """Verify save_output accepts client and model_name parameters."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        from step3_1_GPT4V_video_caption_concise import save_output

        import inspect
        sig = inspect.signature(save_output)
        params = list(sig.parameters.keys())
        self.assertIn("client", params)
        self.assertIn("model_name", params)
        self.assertNotIn("api_key", params)


class TestMainFunction(unittest.TestCase):
    """Test main function signature compatibility."""

    def test_main_accepts_client_and_model(self):
        """Verify main() accepts client and model_name parameters."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        from step3_1_GPT4V_video_caption_concise import main

        import inspect
        sig = inspect.signature(main)
        params = list(sig.parameters.keys())
        self.assertIn("client", params)
        self.assertIn("model_name", params)
        self.assertNotIn("api_key", params)


if __name__ == "__main__":
    unittest.main()

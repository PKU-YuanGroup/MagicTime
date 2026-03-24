"""Integration tests for MiniMax LLM provider.

These tests verify actual API connectivity with MiniMax.
They require MINIMAX_API_KEY to be set in the environment.
Skipped automatically when the key is not available.
"""

import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
SKIP_REASON = "MINIMAX_API_KEY not set"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxTextCompletion(unittest.TestCase):
    """Test MiniMax provider for text-based captioning."""

    def test_minimax_text_completion(self):
        from llm_provider import create_client, get_model_name

        args = argparse.Namespace(
            provider="minimax",
            base_url=None,
            model="MiniMax-M2.7",
            api_key=None,
        )
        client = create_client(args)
        model_name = get_model_name(args)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Summarize in one sentence: A flower blooms from bud to full bloom over 7 days.",
                }
            ],
            max_tokens=128,
        )
        content = response.choices[0].message.content
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 10)

    def test_minimax_call_gpt_function(self):
        """Test call_gpt from step3_1 with MiniMax client."""
        from llm_provider import create_client, get_model_name
        from step3_1_GPT4V_video_caption_concise import call_gpt

        args = argparse.Namespace(
            provider="minimax",
            base_url=None,
            model="MiniMax-M2.7",
            api_key=None,
        )
        client = create_client(args)
        model_name = get_model_name(args)

        prompt = [
            {
                "type": "text",
                "text": "Describe this scene in 20 words: A plant grows from a seed to a small sprout.",
            }
        ]
        result = call_gpt(prompt, client, model_name=model_name)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)

    def test_minimax_call_gpt_detail_function(self):
        """Test call_gpt from step3_1_detail with MiniMax client."""
        from llm_provider import create_client, get_model_name
        from step3_1_GPT4V_video_caption_detail import call_gpt

        args = argparse.Namespace(
            provider="minimax",
            base_url=None,
            model="MiniMax-M2.7",
            api_key=None,
        )
        client = create_client(args)
        model_name = get_model_name(args)

        prompt = [
            {
                "type": "text",
                "text": "Summarize: A candle melts from tall to a small puddle of wax over 3 hours.",
            }
        ]
        result = call_gpt(prompt, client, model_name=model_name)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)


if __name__ == "__main__":
    unittest.main()

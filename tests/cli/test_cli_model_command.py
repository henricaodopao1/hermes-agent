"""Tests for /model slash command UX in the interactive CLI."""

import queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI
from hermes_cli.model_switch import ModelSwitchResult


class TestCliModelCommand:
    def _make_cli(self):
        cli = HermesCLI.__new__(HermesCLI)
        cli.model = "codex/gpt-5.4"
        cli.provider = "openai-codex"
        cli.requested_provider = "openai-codex"
        cli.base_url = "https://chatgpt.com/backend-api/codex"
        cli.api_key = "codex-token"
        cli.api_mode = "responses"
        cli.agent = None
        cli._pending_model_switch_note = None
        cli._explicit_api_key = "codex-token"
        cli._explicit_base_url = "https://chatgpt.com/backend-api/codex"
        cli._model_picker_state = None
        cli._should_exit = False
        cli._invalidate = MagicMock()
        return cli

    def test_bare_model_uses_interactive_picker_instead_of_usage_dump(self):
        cli = self._make_cli()
        model_info = MagicMock()
        model_info.context_window = 256000
        model_info.max_output = 8192
        model_info.has_cost_data.return_value = False
        model_info.format_capabilities.return_value = "tools"

        providers = [
            {
                "slug": "openai-codex",
                "name": "OpenAI Codex",
                "is_current": True,
                "is_user_defined": False,
                "models": ["gpt-5.4", "gpt-5.3-codex"],
                "total_models": 2,
                "source": "hermes",
            },
            {
                "slug": "xai",
                "name": "xAI (Grok)",
                "is_current": False,
                "is_user_defined": False,
                "models": ["grok-4", "grok-3"],
                "total_models": 2,
                "source": "built-in",
            },
        ]

        with patch("hermes_cli.model_switch.parse_model_flags", return_value=("", "", False)), \
             patch("hermes_cli.config.load_config", return_value={}), \
             patch("hermes_cli.model_switch.list_authenticated_providers", return_value=providers), \
             patch.object(cli, "_interactive_model_selection", return_value=("grok-4", "xai")), \
             patch("cli._cprint") as mock_cprint, \
             patch("hermes_cli.model_switch.switch_model", return_value=ModelSwitchResult(
                 success=True,
                 new_model="grok-4",
                 target_provider="xai",
                 provider_changed=True,
                 api_key="xai-token",
                 base_url="https://api.x.ai/v1",
                 provider_label="xAI (Grok)",
                 model_info=model_info,
             )) as mock_switch:
            cli._handle_model_switch("/model")

        mock_switch.assert_called_once_with(
            raw_input="grok-4",
            current_provider="openai-codex",
            current_model="codex/gpt-5.4",
            current_base_url="https://chatgpt.com/backend-api/codex",
            current_api_key="codex-token",
            is_global=False,
            explicit_provider="xai",
            user_providers=None,
        )
        assert cli.model == "grok-4"
        assert cli.provider == "xai"

        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        assert "Aliases:" not in printed
        assert "/model <name>" not in printed
        assert "Model switched: grok-4" in printed

    def test_model_picker_selection_advances_to_model_stage(self):
        cli = self._make_cli()
        state = {
            "stage": "provider",
            "providers": [
                {
                    "slug": "openai-codex",
                    "name": "OpenAI Codex",
                    "is_current": True,
                    "models": ["gpt-5.4"],
                },
                {
                    "slug": "xai",
                    "name": "xAI (Grok)",
                    "is_current": False,
                    "models": ["grok-4", "grok-3"],
                },
            ],
            "choices": ["OpenAI Codex", "xAI (Grok)"],
            "selected": 1,
            "response_queue": queue.Queue(),
        }
        cli._model_picker_state = state

        cli._handle_model_picker_selection()

        assert cli._model_picker_state is state
        assert state["stage"] == "model"
        assert state["provider_slug"] == "xai"
        assert state["choices"] == ["grok-4", "grok-3"]
        assert state["selected"] == 0

    def test_model_picker_selection_emits_final_choice(self):
        cli = self._make_cli()
        response_queue = queue.Queue()
        cli._model_picker_state = {
            "stage": "model",
            "provider_slug": "xai",
            "choices": ["grok-4", "grok-3"],
            "selected": 1,
            "response_queue": response_queue,
        }

        cli._handle_model_picker_selection()

        assert response_queue.get_nowait() == ("grok-3", "xai")

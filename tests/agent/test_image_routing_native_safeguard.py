"""Tests for the native-vision routing safeguard in agent/image_routing.py.

Covers the behaviour added on top of upstream:

* ``image_input_mode: native`` is a *soft* default — when the active model
  is *known* to be text-only (``_lookup_supports_vision`` returns ``False``,
  either via explicit ``supports_vision: false`` override or models.dev),
  the routing falls back to ``"text"`` instead of crashing with a
  provider-side ``unknown variant 'image_url'`` 400.

* Models whose vision capability is unknown (``None``) keep the original
  behaviour: native pass-through, let the provider reject loudly if it
  doesn't accept images. This preserves "try it and see" for custom
  models absent from models.dev.

* The safeguard does NOT change ``auto`` or ``text`` mode behaviour.

Reference: see the user-facing rationale in the patch series
``feature/native-vision-routing-safeguard``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.image_routing import decide_image_input_mode


# ─── Helpers ────────────────────────────────────────────────────────────────


def _cfg(*, image_input_mode: str = "auto", supports_vision=None, provider="minimax"):
    """Build a minimal config dict matching the production layout.

    ``supports_vision``:
      * ``True``/``False`` → written into model.supports_vision (config override)
      * ``None``           → no override (so models.dev is consulted)
    """
    cfg = {"agent": {"image_input_mode": image_input_mode}}
    if supports_vision is not None:
        cfg["model"] = {"provider": provider, "supports_vision": supports_vision}
    return cfg


# ─── Safeguard: native + known-text-only model → text fallback ──────────────


class TestNativeModeKnownTextOnlyFallback:
    """When the user has set image_input_mode: native but the active model
    is *known* to be text-only, route to the text pipeline rather than
    forcing a provider-side error."""

    def test_native_with_explicit_supports_vision_false_falls_back_to_text(self):
        """model.supports_vision: false → routing must defer to text pipeline."""
        cfg = _cfg(image_input_mode="native", supports_vision=False)
        out = decide_image_input_mode("minimax", "MiniMax-M3", cfg)
        assert out == "text"

    def test_native_with_models_dev_text_only_falls_back_to_text(self):
        """When the config has no override and models.dev reports no vision,
        routing must defer to text pipeline (the common case for DeepSeek,
        GPT-OSS, etc.)."""
        cfg = _cfg(image_input_mode="native", supports_vision=None)
        with patch(
            "agent.models_dev.get_model_capabilities",
            return_value=type("Caps", (), {"supports_vision": False})(),
        ):
            out = decide_image_input_mode("deepseek", "deepseek-v4-pro", cfg)
        assert out == "text"

    def test_native_with_models_dev_vision_keeps_native(self):
        """Vision-capable models must keep the native fast path."""
        cfg = _cfg(image_input_mode="native", supports_vision=None)
        with patch(
            "agent.models_dev.get_model_capabilities",
            return_value=type("Caps", (), {"supports_vision": True})(),
        ):
            out = decide_image_input_mode("anthropic", "claude-sonnet-4", cfg)
        assert out == "native"

    def test_native_with_explicit_supports_vision_true_keeps_native(self):
        """Custom models absent from models.dev but declared vision-capable
        must keep the native fast path (e.g. user-added MiniMax-M3)."""
        cfg = _cfg(image_input_mode="native", supports_vision=True)
        out = decide_image_input_mode("minimax", "MiniMax-M3", cfg)
        assert out == "native"


# ─── Unknown capability: preserve existing "try it" behaviour ──────────────


class TestNativeModeUnknownCapability:
    """When capability data is unavailable (None from models.dev and no
    override), the safeguard must NOT downgrade native to text — that would
    silently bypass native vision on every unknown custom model. The
    current behaviour of "try native and let the provider reject loudly"
    is preserved."""

    def test_native_with_unknown_capability_keeps_native(self):
        cfg = _cfg(image_input_mode="native", supports_vision=None)
        with patch(
            "agent.models_dev.get_model_capabilities",
            return_value=None,  # unknown
        ):
            out = decide_image_input_mode("custom", "mystery-model", cfg)
        assert out == "native"

    def test_native_with_lookup_exception_keeps_native(self):
        """Defensive: if capability lookup throws, native still wins so the
        user sees a clear provider error instead of a silent fallback."""
        cfg = _cfg(image_input_mode="native", supports_vision=None)
        with patch(
            "agent.models_dev.get_model_capabilities",
            side_effect=RuntimeError("models.dev down"),
        ):
            out = decide_image_input_mode("custom", "mystery-model", cfg)
        assert out == "native"


# ─── Auto and text modes: behaviour unchanged ───────────────────────────────


class TestAutoAndTextModesUnchanged:
    """The safeguard must not alter auto/text mode behaviour."""

    def test_auto_with_explicit_aux_override_returns_text(self):
        """Existing behaviour: explicit auxiliary.vision.provider → text."""
        cfg = {
            "agent": {"image_input_mode": "auto"},
            "auxiliary": {"vision": {"provider": "xiaomi", "model": "mimo-v2.5"}},
        }
        out = decide_image_input_mode("minimax", "MiniMax-M3", cfg)
        assert out == "text"

    def test_auto_with_vision_capable_model_returns_native(self):
        """Existing behaviour: no aux override + vision model → native."""
        cfg = _cfg(image_input_mode="auto", supports_vision=True)
        out = decide_image_input_mode("minimax", "MiniMax-M3", cfg)
        assert out == "native"

    def test_auto_with_text_only_model_returns_text(self):
        cfg = _cfg(image_input_mode="auto", supports_vision=False)
        out = decide_image_input_mode("deepseek", "deepseek-v4-pro", cfg)
        assert out == "text"

    def test_text_mode_always_returns_text(self):
        """Explicit image_input_mode: text wins over everything."""
        cfg = _cfg(image_input_mode="text", supports_vision=True)
        out = decide_image_input_mode("minimax", "MiniMax-M3", cfg)
        assert out == "text"


# ─── Smoke: provider/model agnosticism ──────────────────────────────────────


class TestProviderAgnostic:
    """The safeguard is a per-turn check — it must not depend on any one
    provider's ID. Switching providers mid-session is normal."""

    @pytest.mark.parametrize(
        "provider,model",
        [
            ("minimax", "MiniMax-M3"),
            ("anthropic", "claude-sonnet-4"),
            ("deepseek", "deepseek-v4-pro"),
            ("openrouter", "google/gemini-2.5-flash"),
            ("xiaomi", "mimo-v2.5-pro"),
        ],
    )
    def test_native_with_text_only_override_always_falls_back(self, provider, model):
        cfg = _cfg(image_input_mode="native", supports_vision=False)
        out = decide_image_input_mode(provider, model, cfg)
        assert out == "text", f"{provider}/{model} should fall back when declared text-only"
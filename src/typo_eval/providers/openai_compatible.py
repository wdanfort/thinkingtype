"""OpenAI-compatible endpoints: Kimi, Together, Fireworks, vLLM, SGLang, ...

Most open-weights hosts expose the OpenAI chat-completions shape, so the whole
adapter is `OpenAIProvider` with a different `base_url` and key. Multi-page
inference (`infer_images`) is inherited unchanged -- hrsim4 items are 2-3 page
documents and every arm goes through it.

Two things differ from the first-party OpenAI path and both matter here:

  * `service_tier` is never sent. It is an OpenAI-only field and compatible
    endpoints reject unknown parameters.
  * `base_url` has no default. Endpoint paths differ per host and a wrong guess
    fails as an auth error rather than a routing error, which is a bad hour to
    spend -- so it must be set explicitly.

Configure via environment:

    KIMI_BASE_URL / KIMI_API_KEY                     -> provider "kimi"
    OPENAI_COMPAT_BASE_URL / OPENAI_COMPAT_API_KEY   -> provider "openai_compatible"
"""

from __future__ import annotations

import os

from openai import OpenAI

from typo_eval.providers.openai import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    """Generic OpenAI-compatible chat-completions endpoint."""

    name = "openai_compatible"
    env_prefix = "OPENAI_COMPAT"
    default_base_url: str | None = None
    key_env_names: tuple = ()  # extra key vars accepted beyond <prefix>_API_KEY

    def __init__(self) -> None:
        base_url = os.environ.get(f"{self.env_prefix}_BASE_URL") or self.default_base_url
        api_key = next(
            (os.environ[n] for n in (f"{self.env_prefix}_API_KEY", *self.key_env_names)
             if os.environ.get(n)),
            None,
        )
        if not base_url:
            raise RuntimeError(
                f"{self.env_prefix}_BASE_URL is not set and provider {self.name!r} has no "
                f"default endpoint -- copy the base URL from the host's API docs "
                f"(it must be the path the OpenAI SDK appends /chat/completions to)."
            )
        if not api_key:
            names = " or ".join((f"{self.env_prefix}_API_KEY", *self.key_env_names))
            raise RuntimeError(f"set {names} for provider {self.name!r}")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=300.0)
        self.last_usage = None
        # OpenAI-only; compatible endpoints reject unknown params. The base class
        # only sends it when truthy, so leaving it None keeps it off the wire.
        self.service_tier = None


class KimiProvider(OpenAICompatibleProvider):
    """Moonshot Kimi -- OpenAI-compatible surface.

    Per the Moonshot docs: base URL https://api.moonshot.ai/v1, model id
    `kimi-k3`, bearer auth from MOONSHOT_API_KEY. Images must be under 4K, and
    hrsim4 renders are 1205x1560, so no provider-side downscaling is expected --
    the smoke test verifies that rather than assuming it. Whether several images
    are accepted in ONE request is not documented; hrsim4 items are 2-3 page
    documents, so the smoke test checks that too.
    """

    name = "kimi"
    env_prefix = "KIMI"
    default_base_url = "https://api.moonshot.ai/v1"
    key_env_names = ("MOONSHOT_API_KEY",)

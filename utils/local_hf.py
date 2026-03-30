from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LocalHFError(RuntimeError):
    pass


def _load_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise LocalHFError(
            "Local HF backend requires 'torch' and 'transformers'. "
            "Install requirements_local_hf.txt first."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFBackendConfig:
    cache_dir: str | None = None
    device_map: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.2


class HFLocalGenerator:
    def __init__(self, config: HFBackendConfig):
        self.config = config
        self._models: dict[str, Any] = {}
        self._tokenizers: dict[str, Any] = {}

    def _load_model(self, model_id: str):
        if model_id in self._models:
            return self._tokenizers[model_id], self._models[model_id]

        torch, AutoModelForCausalLM, AutoTokenizer = _load_transformers()
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
            device_map=self.config.device_map,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model.eval()
        self._tokenizers[model_id] = tokenizer
        self._models[model_id] = model
        return tokenizer, model

    def _build_inputs(self, tokenizer, system_prompt: str, prompt: str):
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            return inputs

        text = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        encoded = tokenizer(text, return_tensors="pt")
        return encoded["input_ids"]

    def generate(self, model_id: str, prompt: str, system_prompt: str) -> str:
        torch, _AutoModelForCausalLM, _AutoTokenizer = _load_transformers()
        tokenizer, model = self._load_model(model_id)

        input_ids = self._build_inputs(tokenizer, system_prompt, prompt)
        if not hasattr(model, "device"):
            raise LocalHFError(f"Loaded model has no .device attribute: {model_id}")
        input_ids = input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=max(1e-5, self.config.temperature),
                pad_token_id=getattr(tokenizer, "eos_token_id", None),
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )

        new_tokens = output_ids[:, input_ids.shape[1] :]
        text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        return text.strip()


def collect_local_model_ids(family: str, local_model_maps: dict[str, dict[str, str]]) -> list[str]:
    families = ["qwen", "llama"] if family == "all" else [family]
    model_ids: list[str] = []
    for fam in families:
        mapping = local_model_maps[fam]
        for domain in ("math", "code", "common"):
            model_ids.append(mapping[domain])
    return sorted(set(model_ids))


def download_local_models(model_ids: list[str], cache_dir: str | None, device_map: str = "auto") -> None:
    _torch, AutoModelForCausalLM, AutoTokenizer = _load_transformers()
    for model_id in model_ids:
        AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=True,
        )


def default_cache_dir(root: Path) -> str:
    return str((root / "models" / "hf_cache").resolve())

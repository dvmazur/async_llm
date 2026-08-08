import functools
import json
import os
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase, AutoProcessor


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def _check_chat_template_inplace(tokenizer: PreTrainedTokenizerBase, model_path):
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass
    return tokenizer

def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    _check_chat_template_inplace(tokenizer, model_path)
    return tokenizer

def load_processor(model_path: str):
    processor = AutoProcessor.from_pretrained(model_path)
    _check_chat_template_inplace(processor.tokenizer, model_path)
    return processor


class _RawConfig:
    """Attribute-accessible wrapper around a raw ``config.json`` dict.

    Fallback for architectures the installed ``transformers`` cannot parse yet
    (e.g. ``qwen3_5``). Nested dicts become ``_RawConfig`` so ``ModelConfig.from_hf``
    can do ``config.text_config.num_attention_heads`` etc.
    """

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, _RawConfig._wrap(v))

    @staticmethod
    def _wrap(v: Any) -> Any:
        if isinstance(v, dict):
            return _RawConfig(**v)
        if isinstance(v, list):
            return [_RawConfig._wrap(x) for x in v]
        return v

    def to_dict(self) -> dict:
        def unwrap(v: Any) -> Any:
            if isinstance(v, _RawConfig):
                return v.to_dict()
            if isinstance(v, list):
                return [unwrap(x) for x in v]
            return v

        return {k: unwrap(v) for k, v in self.__dict__.items()}


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    try:
        return AutoConfig.from_pretrained(model_path)
    except (ValueError, KeyError):
        # Unknown model_type for this transformers version: read raw config.json.
        if os.path.isdir(model_path):
            cfg_file = os.path.join(model_path, "config.json")
        else:
            cfg_file = hf_hub_download(repo_id=model_path, filename="config.json")
        with open(cfg_file, "r", encoding="utf-8") as f:
            return _RawConfig(**json.load(f))


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )

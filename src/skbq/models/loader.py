"""Optional Hugging Face model loading for SKB-Q graph extraction."""

from __future__ import annotations

from dataclasses import dataclass

from skbq.models.operator_mapping import (
    MissingDependencyError,
    detect_supported_architecture,
)


@dataclass(frozen=True, slots=True)
class LoadedHuggingFaceModel:
    """Loaded Hugging Face model bundle."""

    model_name_or_path: str
    model: object
    config: object
    architecture: str


@dataclass(frozen=True, slots=True)
class HuggingFaceModelLoader:
    """Load supported Hugging Face transformer models without graph extraction."""

    local_files_only: bool = True
    trust_remote_code: bool = False

    def load(
        self,
        model_name_or_path: str,
        revision: str | None = None,
        **kwargs: object,
    ) -> LoadedHuggingFaceModel:
        """Load a supported Hugging Face causal transformer model."""

        transformers = _import_transformers()
        config_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        model_kwargs = dict(config_kwargs)
        if revision is not None:
            config_kwargs["revision"] = revision
            model_kwargs["revision"] = revision
        model_kwargs.update(kwargs)

        config = transformers.AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
        architecture = detect_supported_architecture(config)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            **model_kwargs,
        )
        return LoadedHuggingFaceModel(
            model_name_or_path=model_name_or_path,
            model=model,
            config=config,
            architecture=architecture,
        )


def _import_transformers():
    try:
        import transformers
    except ImportError as error:
        raise MissingDependencyError(
            "Hugging Face model loading requires the optional 'transformers' package"
        ) from error
    return transformers

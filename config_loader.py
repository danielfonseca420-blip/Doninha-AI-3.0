"""
Carregamento de configuração centralizada.
==========================================
Lê config.yaml (ou variáveis de ambiente) e expõe um único dicionário
para pipeline, API e agentes.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict

# Diretório raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Carrega a configuração a partir de config.yaml.
    Se o arquivo não existir ou estiver incompleto, usa defaults e env.
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    config: Dict[str, Any] = {
        "knowledge_base": {
            "path": "",
            "chroma_path": os.getenv("VECTOR_DB_PATH", "meu_vector_db"),
            "default_kb": "",
            "domain_specific_kbs": {},
        },
        "l3": {
            "model_path": "truth_scoring_model.pt",
            "backbone": "bert-base-multilingual-cased",
        },
        "l4": {
            "russell_concepts_path": "l4_russell_concepts.json",
        },
        "l4_chain_verification": {
            "provider": os.getenv("L4_COVE_PROVIDER", "template"),
            "ollama_model": os.getenv("L4_COVE_OLLAMA_MODEL", "doninha8:latest"),
            "ollama_host": os.getenv("L4_COVE_OLLAMA_HOST", "http://localhost:11434"),
            "custom_lm_path": "",
        },
        "generation": {
            "provider": os.getenv("GENERATION_PROVIDER", "ollama"),
            "custom_lm_path": "",
            "ollama_model": os.getenv("OLLAMA_MODEL", "doninha8:latest"),
            "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        },
        "finalization": {
            "provider": os.getenv("FINALIZATION_PROVIDER", "ollama"),
            "custom_lm_path": "",
            "ollama_model": os.getenv("FINALIZATION_OLLAMA_MODEL", "doninha8:latest"),
            "ollama_host": os.getenv("FINALIZATION_OLLAMA_HOST", "http://localhost:11434"),
        },
        "l7": {
            "provider": os.getenv("L7_PROVIDER", "ollama"),
            "custom_lm_path": "",
            "model": os.getenv("L7_MODEL", "doninha8:latest"),
            "ollama_host": os.getenv("L7_OLLAMA_HOST", "http://localhost:11434"),
        },
        "agent": {
            "use_agent": os.getenv("USE_AGENT", "false").lower() == "true",
            "vector_db_path": os.getenv("VECTOR_DB_PATH", "meu_vector_db"),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "api": {
            "host": os.getenv("API_HOST", "0.0.0.0"),
            "port": int(os.getenv("API_PORT", "8000")),
        },
        "chat": {
            "max_turns_in_context": 10,
        },
    }

    if path.exists():
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                _deep_merge(config, loaded)
        except Exception:
            pass

    # Resolve paths relativos ao projeto
    for key in ["model_path", "russell_concepts_path", "custom_lm_path", "path", "default_kb"]:
        for section in ["l3", "l4", "l4_chain_verification", "generation", "finalization", "l7", "knowledge_base"]:
            if section in config and key in config[section]:
                val = config[section][key]
                if val and not Path(val).is_absolute():
                    config[section][key] = str(PROJECT_ROOT / val)

    if "knowledge_base" in config and isinstance(config["knowledge_base"].get("domain_specific_kbs"), dict):
        for name, value in config["knowledge_base"]["domain_specific_kbs"].items():
            if value and not Path(value).is_absolute():
                config["knowledge_base"]["domain_specific_kbs"][name] = str(PROJECT_ROOT / value)

    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Mescla override em base recursivamente."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v

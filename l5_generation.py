"""
Camada L5 — Geração de resposta em texto livre.
================================================
A partir da síntese L4 (e contexto L1–L3), gera resposta natural via LLM local (Ollama)
ou fallback para o template da L4. Opcional: LM customizado (EpistemicLanguageModel).
"""

from __future__ import annotations
import os
from typing import Optional

from layer_titles import LAYER_TITLES

# Resultado da L4
try:
    from l4_synthesis import SynthesisResult
except Exception:
    SynthesisResult = None  # type: ignore


def build_context_for_generation(
    prompt: str,
    synthesis_result: "SynthesisResult",
    concepts_summary: str = "",
    top_judgments: str = "",
) -> str:
    """Monta o contexto (texto) a ser enviado ao LLM para gerar a resposta final."""
    lines = [
        "## Contexto epistemológico (L1–L4)",
        f"Pergunta do usuário: {prompt}",
        "",
        f"Resposta sintetizada (L4): {synthesis_result.response}",
        f"Valor de verdade: {synthesis_result.truth_value:.2f} | Estado: {synthesis_result.state} | Certeza: {synthesis_result.certainty:+.2f}",
        "",
        "Use as seguintes nomenclaturas de seção para referenciar as etapas do raciocínio:",
        f"L1: {LAYER_TITLES['l1']}",
        f"L2: {LAYER_TITLES['l2']}",
        f"L3: {LAYER_TITLES['l3']}",
        f"L4: {LAYER_TITLES['l4']}",
        f"L5: {LAYER_TITLES['l5']}",
        f"L6: {LAYER_TITLES['l6']}",
        "",
    ]
    if synthesis_result.supporting_evidence:
        lines.append("Evidências de suporte:")
        for ev in synthesis_result.supporting_evidence[:5]:
            lines.append(f"  - {ev}")
        lines.append("")
    if concepts_summary:
        lines.append(f"{LAYER_TITLES['l1']}: ")
        lines.append(concepts_summary)
        lines.append("")
    if top_judgments:
        lines.append(f"{LAYER_TITLES['l2']}: ")
        lines.append(top_judgments)
        lines.append("")
    lines.append("## Instrução")
    lines.append("Com base no contexto acima, elabore uma resposta final clara e precisa em português, sem repetir literalmente o texto da síntese. Seja conciso e cite a confiança quando relevante.")
    return "\n".join(lines)


def generate_with_ollama(
    context: str,
    model: str = "doninha8:latest",
    ollama_host: str = "http://localhost:11434",
    temperature: float = 0.3,
) -> str:
    """Gera resposta usando Ollama com o modelo local Doninha8."""
    try:
        import os
        import ollama

        if ollama_host:
            os.environ["OLLAMA_HOST"] = ollama_host

        if hasattr(ollama, "chat"):
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": context}],
                stream=False,
                options={
                    "temperature": temperature,
                    "num_ctx": 8192,
                },
            )
            if isinstance(response, dict):
                return response.get("message", {}).get("content", "").strip()
            return str(response).strip() if response else ""

        if hasattr(ollama, "Client"):
            client = ollama.Client(host=ollama_host)
            response = client.generate(
                model=model,
                prompt=context,
                stream=False,
                options={
                    "temperature": temperature,
                    "num_ctx": 8192,
                },
            )
            if isinstance(response, dict):
                return response.get("response", "").strip()
            return str(response).strip() if response else ""

        return ""
    except Exception as e:
        import sys
        print(f"Erro ao gerar com Ollama: {e}", file=sys.stderr)
        return ""


def generate_with_custom_lm(
    context: str,
    model_path: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
) -> str:
    """Gera resposta usando EpistemicLanguageModel (custom_lm_model)."""
    try:
        from custom_lm_model import EpistemicLanguageModel, LMConfig, generate_text, load_lm
        from custom_tokenizer import CustomSPTokenizer, SPConfig
        import torch
        tokenizer = CustomSPTokenizer(SPConfig())
        tokenizer.load()
        vocab_size = tokenizer.vocab_size()
        model = load_lm(model_path, vocab_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        out = generate_text(model, tokenizer, context, max_new_tokens=max_new_tokens, temperature=temperature, device=device)
        return out or ""
    except Exception:
        return ""


def generate_response(
    prompt: str,
    synthesis_result: "SynthesisResult",
    provider: str = "ollama",
    concepts_summary: str = "",
    top_judgments: str = "",
    custom_lm_path: str = "",
    ollama_model: str = "doninha8:latest",
    ollama_host: str = "http://localhost:11434",
) -> str:
    """
    Gera a resposta final em texto livre (ou template).
    provider: "ollama" | "template" | "custom_lm"
    """
    context = build_context_for_generation(prompt, synthesis_result, concepts_summary, top_judgments)

    if provider == "ollama":
        text = generate_with_ollama(context, model=ollama_model, ollama_host=ollama_host, temperature=0.3)
        if text:
            return text.strip()

    if provider == "custom_lm" and custom_lm_path:
        text = generate_with_custom_lm(context, custom_lm_path)
        if text:
            return text.strip()

    # Fallback: resposta da L4 (template)
    return synthesis_result.response

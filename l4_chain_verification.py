"""
Chain of Verification (CoVe) agent para a camada L4.
====================================================
Implementa o workflow Factor + Revise como etapa adicional de verificação
da síntese L4 antes da resposta final ser entregue.
"""

from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Optional, Tuple


def generate_with_ollama(context: str, model: str = "doninha8:latest", ollama_host: str = "http://localhost:11434", temperature: float = 0.2) -> str:
    """Gera resposta usando Ollama com modelo base."""
    try:
        import ollama
        
        if ollama_host:
            os.environ["OLLAMA_HOST"] = ollama_host
        
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
    except Exception:
        return ""


def generate_with_custom_lm(context: str, model_path: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
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


class ChainOfVerificationAgent:
    """Agente de Chain of Verification para a camada L4."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.provider = self.config.get("provider", "template")
        self.ollama_model = self.config.get("ollama_model", "doninha8:latest")
        self.ollama_host = self.config.get("ollama_host", "http://localhost:11434")
        self.custom_lm_path = self.config.get("custom_lm_path", "")

    def verify(
        self,
        prompt: str,
        baseline_response: str,
        context_summary: str,
    ) -> Tuple[str, List[str]]:
        """Executa o workflow CoVe e retorna resposta revisada + log."""
        if self.provider == "ollama":
            output = self._verify_with_ollama(prompt, baseline_response, context_summary)
        elif self.provider == "custom_lm" and self.custom_lm_path:
            output = self._verify_with_custom_lm(prompt, baseline_response, context_summary)
        else:
            output = self._template_verify(prompt, baseline_response, context_summary)

        revised, log = self._parse_verification_output(output, baseline_response)
        return revised, log

    def _verify_with_ollama(self, prompt: str, baseline_response: str, context_summary: str) -> str:
        verification_prompt = self._build_agent_prompt(prompt, baseline_response, context_summary)
        return generate_with_ollama(verification_prompt, model=self.ollama_model, ollama_host=self.ollama_host)

    def _verify_with_custom_lm(self, prompt: str, baseline_response: str, context_summary: str) -> str:
        verification_prompt = self._build_agent_prompt(prompt, baseline_response, context_summary)
        return generate_with_custom_lm(verification_prompt, self.custom_lm_path)

    def _template_verify(self, prompt: str, baseline_response: str, context_summary: str) -> str:
        claims = self._extract_claims(baseline_response)
        questions = self._build_verification_questions(claims)
        verifications = [f"{idx+1}. {q} — Incerto; verificação externa necessária." for idx, q in enumerate(questions)]
        revised = baseline_response.strip()
        if verifications:
            revised += "\n\nNota: esta resposta foi revisada com base em verificação interna limitada; algumas afirmações permanecem pendentes de confirmação externa."
        sections = [
            "Baseline Response:",
            baseline_response.strip(),
            "",
            "Verification Questions:",
            *questions,
            "",
            "Independent Verification Results:",
            *verifications,
            "",
            "Cross-Check & Revise:",
            "Nenhuma inconsistência formal identificada no conteúdo disponível localmente.",
            "",
            "Revised Response:",
            revised,
        ]
        return "\n".join(sections)

    def _build_agent_prompt(self, prompt: str, baseline_response: str, context_summary: str) -> str:
        lines = [
            "Você é um engenheiro de prompts especialista em técnicas avançadas de confiabilidade.",
            "A partir de agora, use o método Chain of Verification (CoVe) - variante Factor + Revise para analisar e revisar a resposta.",
            "Responda usando sempre o fluxo: 1. Baseline Response, 2. Factoring, 3. Independent Verification, 4. Cross-Check & Revise.",
            "Seja rigoroso, conservador, e declare limitações quando necessário.",
            "",
            f"Pergunta original: {prompt}",
            "",
            "Contexto resumido de L4: ",
            context_summary or "Sem contexto adicional disponível.",
            "",
            "Resposta inicial (Baseline Response):",
            baseline_response.strip(),
            "",
            "Tarefa:",
            "1. Gere de 6 a 12 perguntas de verificação independentes a partir das principais afirmações da resposta inicial.",
            "2. Responda cada pergunta de forma independente, marcando como Confirmado, Refutado, Parcialmente correto ou Incerto.",
            "3. Compare a resposta inicial com os resultados e reescreva a resposta final incorporando apenas o que foi verificado.",
            "4. Entregue a estrutura completa com as seções claramente demarcadas e finalize com a resposta revisada.",
            "",
            "Formato de saída exigido:",
            "Baseline Response:",
            "<texto>",
            "",
            "Verification Questions:",
            "1. <pergunta>",
            "...",
            "",
            "Independent Verification Results:",
            "1. <marcação> — <resposta>",
            "...",
            "",
            "Cross-Check & Revise:",
            "<análise>",
            "",
            "Revised Response:",
            "<texto revisado>",
        ]
        return "\n".join(lines)

    def _parse_verification_output(self, output: str, baseline_response: str) -> Tuple[str, List[str]]:
        if not output:
            return baseline_response, ["Nenhuma saída de verificação gerada."]

        revised = baseline_response
        log: List[str] = []
        if "Revised Response:" in output:
            parts = output.split("Revised Response:")
            revised = parts[-1].strip()
            log = [line.strip() for line in output.splitlines() if line.strip()]
        else:
            log = [line.strip() for line in output.splitlines() if line.strip()]
        return revised, log

    def _extract_claims(self, text: str) -> List[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\\s+", text) if s.strip()]
        claims = []
        for sentence in sentences:
            if len(claims) >= 12:
                break
            if len(sentence.split()) >= 5:
                claims.append(sentence)
        return claims[:12] if claims else sentences[:min(6, len(sentences))]

    def _build_verification_questions(self, claims: List[str]) -> List[str]:
        questions: List[str] = []
        for claim in claims[:12]:
            question = f"A afirmação a seguir está correta e fundamentada? {claim}"
            questions.append(question)
        if len(questions) < 6:
            questions.extend([
                "A estrutura lógica da resposta está consistente com a informação disponível?",
                "Há alguma suposição implícita que precisa ser explicitada ou verificada?",
            ])
        return questions[:12]

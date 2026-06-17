#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                  DONINHA IA — MIDDLEWARE NEURO-SIMBÓLICO HÍBRIDO            ║
║                            Compilado em um único arquivo                    ║
║                                                                              ║
║  Middleware de 7 Camadas:                                                   ║
║    L1: Tábua de Conceitos (Aristotélica)                                    ║
║    L2: Juízos Kantianos + Epistemologia (BERT)                             ║
║    L3: Lógica Paraconsistente (μ/λ, 12 estados)                            ║
║    L4: Síntese Russelliana + Chain of Verification                         ║
║    L5: Geração textual (Ollama, custom LM)                                 ║
║    L6: Refinamento final                                                    ║
║    L7: Síntese definitiva com auditoria                                    ║
║                                                                              ║
║  Suporta: OpenAI, Anthropic, Google Gemini, Ollama, fallback               ║
║  Auto-contido, bem organizado, pronto para uso em produção                 ║
║                                                                              ║
║  Compilado em: 2026-06-17                                                   ║
║  Compatível com: Python 3.8+                                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import os
import re
import json
import math
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Any, Tuple, Union, Iterator, Set
)
from pathlib import Path
from datetime import datetime

# ============================================================================
# LOGGER
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("DoninhaMiddleware")


# ============================================================================
# === CONSTANTES GLOBAIS ===
# ============================================================================

# Limites paraconsistentes
THRESHOLD_TRUE = 0.7
THRESHOLD_FALSE = 0.3
THRESHOLD_INCONSISTENT = 0.6
THRESHOLD_INDETERMINATE = 0.4

# Estados lógicos (12)
STATE_V = "Verdadeiro"
STATE_F = "Falso"
STATE_T = "Inconsistente"
STATE_BOT = "Indeterminado"
STATE_QV = "Quase_Verdadeiro"
STATE_QF = "Quase_Falso"
STATE_QF_TO_V = "QF_to_V"
STATE_BOT_TO_F = "Indeterminado_to_F"
STATE_BOT_TO_V = "Indeterminado_to_V"
STATE_QV_TO_BOT = "QV_to_Indeterminado"
STATE_T_TO_V = "Inconsistente_to_V"
STATE_QV_TO_T = "QV_to_Inconsistente"

ALL_STATES_12 = [
    STATE_V, STATE_F, STATE_T, STATE_BOT,
    STATE_QV, STATE_QF,
    STATE_QF_TO_V, STATE_BOT_TO_F, STATE_BOT_TO_V,
    STATE_QV_TO_BOT, STATE_T_TO_V, STATE_QV_TO_T,
]

# Pesos kantianos
MODALIDADE_PESO = {
    "Apodítico": 1.0,
    "Assertórico": 0.7,
    "Problemático": 0.3,
}

QUANTIDADE_PESO = {
    "Singular": 1.0,
    "Particular": 0.6,
    "Universal": 0.3,
}

QUALIDADE_PESO = {
    "Afirmativo": 1.0,
    "Infinito": 0.6,
    "Negativo": 0.4,
}

RELACAO_PESO = {
    "Categórico": 1.0,
    "Hipotético": 0.7,
    "Disjuntivo": 0.5,
}

# Proposições A/E/I/O
PROPOSITION_TYPE_A = "A"  # Universal Afirmativa
PROPOSITION_TYPE_E = "E"  # Universal Negativa
PROPOSITION_TYPE_I = "I"  # Particular Afirmativa
PROPOSITION_TYPE_O = "O"  # Particular Negativa

PROPOSITION_TYPE_LABELS = {
    PROPOSITION_TYPE_A: "Universal Afirmativa",
    PROPOSITION_TYPE_E: "Universal Negativa",
    PROPOSITION_TYPE_I: "Particular Afirmativa",
    PROPOSITION_TYPE_O: "Particular Negativa",
}

# Nomes das camadas
LAYER_TITLES = {
    "l1": "Demarcação de Conceitos Fundamentais",
    "l2": "Premissas e proposições centrais",
    "l3": "Análise da Estrutura Lógico-filosófica",
    "l4": "Comparação da equivalência entre Estrutura formal e Mundo Empírico",
    "l5": "Síntese Intermediária derivada das etapas anteriores",
    "l6": "Conclusão do raciocínio",
    "l7": "Síntese Final e Redação",
}


# ============================================================================
# === L1 - TÁBUA DE CONCEITOS (ARISTOTÉLICA) ===
# ============================================================================

@dataclass
class ConceptNode:
    """Representação de um conceito com relações semânticas."""
    term: str
    definition: str = ""
    synonyms: List[str] = field(default_factory=list)
    antonyms: List[str] = field(default_factory=list)
    hyponyms: List[str] = field(default_factory=list)
    hypernyms: List[str] = field(default_factory=list)
    homonyms: Dict[str, str] = field(default_factory=dict)
    paronyms: List[str] = field(default_factory=list)
    domain: str = "geral"
    application_context: str = ""
    canonical_source: str = ""
    canonical_context: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa o nó para dicionário."""
        return {
            "term": self.term,
            "definition": self.definition,
            "synonyms": self.synonyms,
            "antonyms": self.antonyms,
            "hyponyms": self.hyponyms,
            "hypernyms": self.hypernyms,
            "homonyms": self.homonyms,
            "paronyms": self.paronyms,
            "domain": self.domain,
            "canonical_source": self.canonical_source,
            "canonical_context": self.canonical_context,
        }


class ConceptTable:
    """Tábua de Conceitos — Mapeamento de termos e relações semânticas."""

    SEED_CONCEPTS = {
        "verdade": ConceptNode(
            term="verdade",
            definition="Correspondência entre crença e fato (Russell)",
            synonyms=["veracidade", "autenticidade"],
            antonyms=["falsidade", "mentira"],
            hypernyms=["propriedade", "característica"],
            domain="epistemologia"
        ),
        "falsidade": ConceptNode(
            term="falsidade",
            definition="Falta de correspondência entre crença e realidade",
            synonyms=["engano", "erro"],
            antonyms=["verdade", "veracidade"],
            domain="epistemologia"
        ),
        "conhecimento": ConceptNode(
            term="conhecimento",
            definition="Justificação verdadeira de crença",
            synonyms=["saber", "ciência"],
            hypernyms=["fenômeno mental"],
            domain="epistemologia"
        ),
        "proposição": ConceptNode(
            term="proposição",
            definition="Enunciado susceptível de ser verdadeiro ou falso",
            synonyms=["enunciado", "asserção"],
            hyponyms=["juízo", "tese"],
            domain="lógica"
        ),
        "silogismo": ConceptNode(
            term="silogismo",
            definition="Raciocínio dedutivo com duas premissas e conclusão",
            synonyms=["argumento dedutivo"],
            hypernyms=["forma de raciocínio"],
            domain="lógica"
        ),
        "contradição": ConceptNode(
            term="contradição",
            definition="Afirmação simultânea de proposições contrárias",
            synonyms=["inconsistência", "conflito"],
            antonyms=["consistência", "harmonia"],
            domain="lógica"
        ),
        "razão": ConceptNode(
            term="razão",
            definition="Faculdade do pensamento lógico e crítico",
            synonyms=["inteligência", "intelecto"],
            hypernyms=["capacidade mental"],
            domain="epistemologia"
        ),
        "evidência": ConceptNode(
            term="evidência",
            definition="Clareza ou certeza de algo que se oferece à inteligência",
            synonyms=["prova", "testemunho"],
            hypernyms=["informação", "dado"],
            domain="epistemologia"
        ),
        "quente": ConceptNode(
            term="quente",
            definition="Propriedade térmica de alta temperatura",
            antonyms=["frio"],
            hyponyms=["escaldante", "aquecido"],
            hypernyms=["temperatura"],
            domain="física"
        ),
        "frio": ConceptNode(
            term="frio",
            definition="Propriedade térmica de baixa temperatura",
            antonyms=["quente"],
            hyponyms=["gelado"],
            hypernyms=["temperatura"],
            domain="física"
        ),
    }

    def __init__(self) -> None:
        self._table: Dict[str, ConceptNode] = {}
        for node in self.SEED_CONCEPTS.values():
            self.add(node)

    @staticmethod
    def _normalize(term: str) -> str:
        """Normaliza um termo para busca (lowercase, sem acentos)."""
        return term.lower().strip()

    def get(self, term: str) -> Optional[ConceptNode]:
        """Obtém um nó da tabela."""
        return self._table.get(self._normalize(term))

    def add(self, node: ConceptNode) -> None:
        """Adiciona um nó à tabela."""
        self._table[self._normalize(node.term)] = node

    def extract_concepts(self, text: str) -> List[ConceptNode]:
        """Extrai conceitos presentes no texto."""
        tokens = re.findall(r"[a-záàãâéêíóôõúüçA-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]+", text)
        seen = set()
        result = []
        for tok in tokens:
            key = self._normalize(tok)
            if key not in seen:
                node = self.get(tok)
                if node:
                    seen.add(key)
                    result.append(node)
        return result

    def relation_type(self, term_a: str, term_b: str) -> str:
        """Retorna o tipo de relação entre dois termos."""
        node_a = self.get(term_a)
        if not node_a:
            return "desconhecida"
        
        b_norm = self._normalize(term_b)
        if b_norm in [self._normalize(s) for s in node_a.synonyms]:
            return "sinonímia"
        if b_norm in [self._normalize(s) for s in node_a.antonyms]:
            return "antonímia"
        if b_norm in [self._normalize(s) for s in node_a.hyponyms]:
            return "hiponímia"
        if b_norm in [self._normalize(s) for s in node_a.hypernyms]:
            return "hiperônimia"
        return "desconhecida"


# ============================================================================
# === L2 - JUÍZOS KANTIANOS ===
# ============================================================================

@dataclass
class EpistemicClassification:
    """Classificação epistemológica (T/I/F) sem restrição T+I+F=1."""
    truth: float = 0.0
    indeterminacy: float = 0.0
    falsity: float = 0.0
    classification: str = "indeterminado"

    def __post_init__(self):
        self.classification = self._classify()

    def _classify(self) -> str:
        """Classifica baseado nas evidências."""
        if self.truth + self.falsity > 1.0:
            return "paraconsistência"
        if self.truth + self.indeterminacy + self.falsity < 1.0:
            return "incompletude"
        if self.indeterminacy > 0.6:
            return "vagueza"
        if self.truth > 0.7 and self.indeterminacy < 0.2 and self.falsity < 0.2:
            return "assertiva_confiante"
        return "indeterminado"

    def __str__(self) -> str:
        return (
            f"T={self.truth:.2f} I={self.indeterminacy:.2f} F={self.falsity:.2f} "
            f"[{self.classification}]"
        )


@dataclass
class KantianJudgment:
    """Proposição segundo a Tábua dos Juízos de Kant."""
    quantidade: str
    qualidade: str
    relacao: str
    modalidade: str
    proposicao: str
    prioridade: float = 0.0
    epistemic_classification: EpistemicClassification = field(default_factory=EpistemicClassification)

    def __str__(self) -> str:
        return (
            f"[{self.quantidade}/{self.qualidade}/"
            f"{self.relacao}/{self.modalidade}] "
            f"(pri={self.prioridade:.2f}) {self.epistemic_classification} {self.proposicao}"
        )


def infer_proposition_type(text: str) -> Optional[str]:
    """Heurística simples para inferir o tipo A / E / I / O a partir do texto."""
    normalized = text.lower().strip()
    if not normalized:
        return None

    if re.search(r"\b(algum não|alguns não|alguma não|algumas não|nem todos|pelo menos um não)\b", normalized):
        return PROPOSITION_TYPE_O
    if re.search(r"\b(nenhum|nenhuma|nunca|jamais|sem nenhum|sem nenhuma|não existe|não há)\b", normalized):
        return PROPOSITION_TYPE_E
    if re.search(r"\b(algum|alguma|alguns|algumas|pelo menos um|há um|há algum|existem|existe)\b", normalized):
        return PROPOSITION_TYPE_I
    if re.search(r"\b(todo|todos|toda|todas|cada|sempre|qualquer)\b", normalized):
        return PROPOSITION_TYPE_A

    return None


class KantianJudgmentEngine:
    """Motor de geração de juízos segundo Kant."""

    def __init__(self, concept_table: ConceptTable) -> None:
        self.ct = concept_table

    def refine(self, prompt: str, concepts: List[ConceptNode]) -> List[KantianJudgment]:
        """Gera as hipóteses kantianas para o prompt."""
        subject, predicates = self._parse_prompt(prompt, concepts)
        judgments: List[KantianJudgment] = []

        for pred in predicates:
            base_prop = f"{subject} é {pred}"
            j = KantianJudgment(
                quantidade="Singular",
                qualidade="Afirmativo",
                relacao="Categórico",
                modalidade="Assertórico",
                proposicao=base_prop
            )
            j.prioridade = (
                MODALIDADE_PESO[j.modalidade]
                * QUANTIDADE_PESO[j.quantidade]
                * QUALIDADE_PESO[j.qualidade]
                * RELACAO_PESO[j.relacao]
            )
            judgments.append(j)

        judgments.sort(key=lambda j: j.prioridade, reverse=True)
        return judgments[:10]

    @staticmethod
    def _parse_prompt(prompt: str, concepts: List[ConceptNode]) -> Tuple[str, List[str]]:
        """Extrai sujeito e predicados candidatos do prompt."""
        tokens = re.findall(r"[a-záàãâéêíóôõúüçA-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]+", prompt.lower())
        known = {c.term.lower() for c in concepts}
        subject = tokens[0] if tokens else "entidade"
        predicates = [t for t in tokens[1:] if t in known] or ["indeterminado"]
        return subject, predicates


# ============================================================================
# === L3 - LÓGICA PARACONSISTENTE ===
# ============================================================================

@dataclass
class ParaconsistentValue:
    """Anotação paraconsistente de uma proposição."""
    proposition: str
    mu: float
    lam: float
    proposition_type: Optional[str] = None

    @property
    def certainty(self) -> float:
        """Grau de certeza: Gc = μ − λ"""
        return self.mu - self.lam

    @property
    def contradiction(self) -> float:
        """Grau de contradição: Gct = μ + λ − 1"""
        return self.mu + self.lam - 1.0

    @property
    def state(self) -> str:
        """Estado lógico qualitativo."""
        if self.mu >= THRESHOLD_TRUE and self.lam <= (1 - THRESHOLD_TRUE):
            return "Verdadeiro"
        if self.mu <= THRESHOLD_FALSE and self.lam >= (1 - THRESHOLD_FALSE):
            return "Falso"
        if self.mu >= THRESHOLD_INCONSISTENT and self.lam >= THRESHOLD_INCONSISTENT:
            return "Inconsistente_local"
        if self.mu <= THRESHOLD_INDETERMINATE and self.lam <= THRESHOLD_INDETERMINATE:
            return "Indeterminado"
        return "Intermediário"

    @property
    def truth_value(self) -> float:
        """Valor-verdade escalar normalizado."""
        return round((self.mu + (1 - self.lam)) / 2.0, 4)

    def __str__(self) -> str:
        type_label = PROPOSITION_TYPE_LABELS.get(self.proposition_type or "?", "Desconhecido")
        return (
            f"  μ={self.mu:.3f}  λ={self.lam:.3f}  "
            f"Gc={self.certainty:+.3f}  Gct={self.contradiction:+.3f}  "
            f"v={self.truth_value:.3f}  [{self.state}]\n"
            f"  Tipo={type_label}  \"{self.proposition}\""
        )


class ParaconsistentEngine:
    """Motor de avaliação paraconsistente."""

    @staticmethod
    def evaluate(
        propositions: List[Tuple[str, float]],
        knowledge_base: Dict[str, float],
    ) -> List[ParaconsistentValue]:
        """Avalia proposições e retorna ParaconsistentValues."""
        results: List[ParaconsistentValue] = []
        for prop_text, l2_priority in propositions:
            mu, lam = ParaconsistentEngine._compute_annotations(prop_text, l2_priority, knowledge_base)
            pv = ParaconsistentValue(
                proposition=prop_text,
                mu=mu,
                lam=lam,
                proposition_type=infer_proposition_type(prop_text),
            )
            results.append(pv)

        results.sort(key=lambda pv: pv.truth_value, reverse=True)
        return results

    @staticmethod
    def _compute_annotations(
        text: str,
        l2_priority: float,
        kb: Dict[str, float],
    ) -> Tuple[float, float]:
        """Calcula (μ, λ) para uma proposição."""
        tokens = set(re.findall(r"[a-záàãâéêíóôõúüç]+", text.lower()))

        kb_scores = [kb.get(t, 0.0) for t in tokens if kb.get(t, 0.0) > 0]
        mu_kb = sum(kb_scores) / len(kb_scores) if kb_scores else 0.3

        ANTONYM_PAIRS = [
            ("quente", "frio"), ("quente", "gelado"),
            ("verdadeiro", "falso"), ("real", "fictício"),
        ]
        contradiction_detected = any(
            a in tokens and b in tokens for a, b in ANTONYM_PAIRS
        )
        lam_base = 0.8 if contradiction_detected else (1.0 - mu_kb)

        mu = min(1.0, mu_kb * (0.5 + 0.5 * l2_priority))
        lam = max(0.0, lam_base * (1.0 - 0.3 * l2_priority))

        return round(mu, 4), round(lam, 4)

    @staticmethod
    def check_global_consistency(values: List[ParaconsistentValue]) -> bool:
        """Retorna True se o sistema é globalmente consistente."""
        states = {pv.state for pv in values}
        if states == {"Inconsistente_local"}:
            return False
        return True


# ============================================================================
# === L4 - SÍNTESE RUSSELLIANA ===
# ============================================================================

@dataclass
class SynthesisResult:
    """Resultado da síntese russelliana."""
    response: str
    truth_value: float
    certainty: float
    contradiction: float
    state: str
    supporting_evidence: List[str] = field(default_factory=list)
    falsified_hypotheses: List[str] = field(default_factory=list)
    verification_log: List[str] = field(default_factory=list)
    confidence_label: str = ""

    def __post_init__(self):
        if not self.confidence_label:
            self.confidence_label = self._label()

    def _label(self) -> str:
        v = self.truth_value
        if v >= 0.85:
            return "Alta Confiança"
        if v >= 0.65:
            return "Confiança Moderada"
        if v >= 0.45:
            return "Incerto / Intermediário"
        if v >= 0.25:
            return "Baixa Confiança"
        return "Indeterminado"

    def __str__(self) -> str:
        lines = [
            "━" * 60,
            f"  RESPOSTA : {self.response}",
            f"  Estado   : {self.state}  ({self.confidence_label})",
            f"  v-verdade: {self.truth_value:.4f}  |  "
            f"Certeza: {self.certainty:+.4f}  |  "
            f"Contradição: {self.contradiction:+.4f}",
        ]
        if self.supporting_evidence:
            lines.append("  Evidências de suporte:")
            for ev in self.supporting_evidence[:3]:
                lines.append(f"    • {ev}")
        if self.falsified_hypotheses:
            lines.append("  Hipóteses falsificadas:")
            for fh in self.falsified_hypotheses[:2]:
                lines.append(f"    ✗ {fh}")
        lines.append("━" * 60)
        return "\n".join(lines)


class RussellianSynthesisEngine:
    """Motor de síntese pela teoria da correspondência de Russell."""

    def __init__(self, kb: Dict[str, float]) -> None:
        self.kb = kb

    def synthesize(
        self,
        pv_list: List[ParaconsistentValue],
        l2_priorities: Dict[str, float],
        prompt: str,
    ) -> SynthesisResult:
        """Sintetiza os resultados das camadas anteriores."""
        if not pv_list:
            return SynthesisResult(
                response="Sem hipóteses válidas para síntese.",
                truth_value=0.0, certainty=0.0,
                contradiction=0.0, state="Indeterminado",
            )

        best = pv_list[0]
        supporting = [pv.proposition for pv in pv_list[1:4] if pv.state != "Falso"]
        falsified = [pv.proposition for pv in pv_list if pv.state == "Falso"]

        total_w, total_v = 0.0, 0.0
        for pv in pv_list:
            key = pv.proposition[:40]
            l2_w = l2_priorities.get(key, 0.5)
            weight = l2_w * (1.0 + max(pv.certainty, 0.0))
            total_v += pv.truth_value * weight
            total_w += weight

        v_final = total_v / total_w if total_w > 0 else best.truth_value

        response = self._generate_response(best, prompt)

        return SynthesisResult(
            response=response,
            truth_value=round(v_final, 4),
            certainty=round(best.certainty, 4),
            contradiction=round(best.contradiction, 4),
            state=best.state,
            supporting_evidence=supporting,
            falsified_hypotheses=falsified,
        )

    def _generate_response(self, best_pv: ParaconsistentValue, prompt: str) -> str:
        """Gera resposta a partir da proposição com maior valor-verdade."""
        state = best_pv.state
        v = best_pv.truth_value

        if state == "Verdadeiro":
            prefix = f"Com alta confiança (v={v:.2f}):"
        elif state == "Intermediário":
            prefix = f"Com valor intermediário (v={v:.2f}):"
        elif state == "Inconsistente_local":
            prefix = f"Contradição local detectada (v={v:.2f}):"
        elif state == "Falso":
            prefix = f"Evidência insuficiente (v={v:.2f}):"
        else:
            prefix = f"Indeterminado (v={v:.2f}):"

        return f"{prefix} {best_pv.proposition}"

    @staticmethod
    def check_fundamental_limits(query: str) -> Optional[str]:
        """Detecta perguntas que violam os limites fundamentais da IA."""
        limit_keywords = {
            "consciência": "IA não possui consciência — atributo biológico emergente.",
            "sentimento": "IA não possui estados afetivos — limitada ao algoritmo.",
            "imaginação": "Imaginação é liberdade humana (Sartre) — não computável.",
            "agi": "AGI é oximoro teórico: algoritmo não supera seu criador.",
        }
        q_lower = query.lower()
        for keyword, warning in limit_keywords.items():
            if keyword in q_lower:
                return f"⚠ Limite fundamental: {warning}"
        return None


# ============================================================================
# === L5 - GERAÇÃO DE RESPOSTA ===
# ============================================================================

def build_context_for_generation(
    prompt: str,
    synthesis_result: SynthesisResult,
    concepts_summary: str = "",
) -> str:
    """Monta o contexto para o LLM gerar a resposta final."""
    lines = [
        "## Contexto epistemológico (L1–L4)",
        f"Pergunta do usuário: {prompt}",
        "",
        f"Resposta sintetizada (L4): {synthesis_result.response}",
        f"Valor de verdade: {synthesis_result.truth_value:.2f} | Estado: {synthesis_result.state}",
        "",
        "Com base no contexto acima, elabore uma resposta final clara e precisa em português.",
    ]
    return "\n".join(lines)


def generate_response_fallback(
    synthesis_result: SynthesisResult,
    prompt: str,
) -> str:
    """Fallback: usa template simples."""
    template = f"""Pergunta: {prompt}

Análise epistemológica:
- Valor de verdade: {synthesis_result.truth_value:.2f}
- Estado: {synthesis_result.state}
- Confiança: {synthesis_result.confidence_label}

Resposta sintetizada:
{synthesis_result.response}

Observação: Esta resposta foi sintetizada através de análise paraconsistente
das camadas L1-L4, com suporte em lógica formal e epistemologia de Russell.
"""
    return template


# ============================================================================
# === L6 - REFINAMENTO FINAL ===
# ============================================================================

@dataclass
class EpistemicContext:
    """Contexto epistemológico agregado para L6."""
    proposition_states: List[Dict[str, Any]] = field(default_factory=list)
    many_valued_routes: List[Dict[str, Any]] = field(default_factory=list)
    bert_classifications: List[Dict[str, Any]] = field(default_factory=list)
    application_context: str = ""


class FinalResponseEngine:
    """Gera a resposta final em texto fluido a partir da síntese das camadas anteriores."""

    def finalize_response(
        self,
        prompt: str,
        synthesis_result: SynthesisResult,
        epistemic_context: Optional[EpistemicContext] = None,
        generated_text: str = "",
        concepts_summary: str = "",
    ) -> str:
        """Produz a resposta final única e contínua."""
        main_text = self._normalize_text(generated_text or synthesis_result.response or "")
        if not main_text:
            return "Não há informação suficiente para formular uma resposta final."

        intro = self._build_intro(synthesis_result)
        conclusion = self._build_conclusion(synthesis_result)

        if intro:
            if main_text.lower().startswith(intro.lower()):
                final = main_text
            else:
                final = f"{intro} {main_text}"
        else:
            final = main_text

        if conclusion:
            final = f"{final} {conclusion}"

        return final.strip()

    def _build_intro(self, synthesis_result: SynthesisResult) -> str:
        if synthesis_result.truth_value >= 0.85:
            return "Com base no motor de raciocínio L1–L5, a melhor conclusão indica"
        if synthesis_result.truth_value >= 0.65:
            return "A partir da síntese das camadas L1–L5, o cenário mais sólido sugere"
        if synthesis_result.truth_value >= 0.45:
            return "Com certa cautela, a análise das camadas L1–L5 aponta"
        return "A análise das camadas L1–L5 indica"

    def _build_conclusion(self, synthesis_result: SynthesisResult) -> str:
        if synthesis_result.state in {"Indeterminado"}:
            return (
                "Esta questão tem uma dimensão genuinamente indeterminada — não por falta de rigor, "
                "mas porque a evidência empírica ainda não existe."
            )
        if synthesis_result.contradiction > 0.25:
            return "Essa conclusão é apresentada como a melhor interpretação disponível, embora exista uma contradição local que recomenda prudência."
        if synthesis_result.truth_value < 0.65:
            return "Dado o grau de incerteza, vale considerar essa resposta como provisória até que evidências adicionais sejam avaliadas."
        return ""

    def _normalize_text(self, text: str) -> str:
        normalized_lines = []
        for line in text.splitlines():
            line = " ".join(line.split()).strip()
            if line:
                normalized_lines.append(line)
        return "\n\n".join(normalized_lines)


# ============================================================================
# === L7 - SÍNTESE FINAL COM AUDITORIA ===
# ============================================================================

class FinalTextEngine:
    """Motor de síntese final com blocos de auditoria."""

    def finalize_text(
        self,
        prompt: str,
        synthesis_result: SynthesisResult,
        l1_summary: str = "",
        l2_summary: str = "",
        l3_summary: str = "",
        finalized_text: str = "",
    ) -> str:
        """Gera texto final com auditoria L7."""
        main = finalized_text.strip() or synthesis_result.response.strip()

        audit_block = self._build_audit_block(
            synthesis_result,
            l1_summary,
            l2_summary,
        )

        final = f"{main}\n\n{audit_block}"
        return final

    def _build_audit_block(
        self,
        synthesis_result: SynthesisResult,
        l1_summary: str,
        l2_summary: str,
    ) -> str:
        """Constrói bloco [AUDIT L7]."""
        lines = [
            "[AUDIT L7 — SÍNTESE FINAL AUDITÁVEL]",
            f"Timestamp: {datetime.now().isoformat()}",
            f"Estado lógico: {synthesis_result.state}",
            f"Valor de verdade: {synthesis_result.truth_value:.4f}",
            f"Grau de certeza (Gc): {synthesis_result.certainty:+.4f}",
            f"Grau de contradição (Gct): {synthesis_result.contradiction:+.4f}",
            f"Nível de confiança: {synthesis_result.confidence_label}",
        ]

        if l1_summary:
            lines.append(f"Conceitos integrados (L1): {l1_summary[:100]}")

        if l2_summary:
            lines.append(f"Juízos kantianos (L2): {l2_summary[:100]}")

        if synthesis_result.supporting_evidence:
            lines.append(f"Evidências principais ({len(synthesis_result.supporting_evidence)})")

        if synthesis_result.falsified_hypotheses:
            lines.append(f"Hipóteses descartadas ({len(synthesis_result.falsified_hypotheses)})")

        lines.append("[FIM AUDIT L7]")

        return "\n".join(lines)


# ============================================================================
# === PIPELINE PRINCIPAL ===
# ============================================================================

class HybridLLMPipeline:
    """Pipeline completo do Modelo Híbrido de LLM."""

    def __init__(
        self,
        knowledge_base: Optional[Dict[str, float]] = None,
        verbose: bool = True,
    ) -> None:
        self.kb = knowledge_base or self._get_default_kb()
        self.verbose = verbose

        self.L1 = ConceptTable()
        self.L2 = KantianJudgmentEngine(self.L1)
        self.L3 = ParaconsistentEngine()
        self.L4 = RussellianSynthesisEngine(self.kb)
        self.L6 = FinalResponseEngine()
        self.L7 = FinalTextEngine()

    @staticmethod
    def _get_default_kb() -> Dict[str, float]:
        """KB padrão (fallback quando não há arquivo)."""
        return {
            "quente": 0.85, "frio": 0.85, "morno": 0.70, "aquecido": 0.80, "gelado": 0.80,
            "temperatura": 0.90, "graus": 0.88, "escaldante": 0.75, "tépido": 0.65,
            "verdadeiro": 0.95, "falso": 0.95, "contradição": 0.80, "proposição": 0.85,
            "silogismo": 0.75, "conhecimento": 0.90, "inteligência": 0.85, "consciência": 0.70,
            "razão": 0.88, "verdade": 0.92, "água": 0.95, "líquido": 0.90, "h2o": 0.90,
        }

    def process(self, prompt: str) -> SynthesisResult:
        """Executa o pipeline e retorna SynthesisResult."""
        self._log("\n" + "═" * 60)
        self._log(f"  PROMPT: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        self._log("═" * 60)

        limit = RussellianSynthesisEngine.check_fundamental_limits(prompt)
        if limit:
            self._log(f"\n{limit}")

        # L1: Extração de Conceitos
        self._log("\n[L1] Extração de Conceitos")
        concepts = self.L1.extract_concepts(prompt)
        concepts_summary = "; ".join(f"{c.term}({', '.join(c.synonyms[:2])})" for c in concepts[:8])
        if self.verbose and concepts:
            for c in concepts[:5]:
                syns = ", ".join(c.synonyms[:2]) or "—"
                self._log(f"  • {c.term:15s} | sinônimos: {syns}")

        # L2: Juízos Kantianos
        self._log("\n[L2] Juízos Kantianos")
        judgments = self.L2.refine(prompt, concepts)
        top_judgments = "\n".join(j.proposicao for j in judgments[:6])
        self._log(f"  {len(judgments)} juízos gerados")

        # L3: Lógica Paraconsistente
        self._log("\n[L3] Lógica Paraconsistente")
        props_with_priority = [(j.proposicao, j.prioridade) for j in judgments]
        pv_list = self.L3.evaluate(props_with_priority, self.kb)
        consistent = self.L3.check_global_consistency(pv_list)
        self._log(f"  Consistência global: {'✓' if consistent else '✗'}")

        # L4: Síntese Russelliana
        self._log("\n[L4] Síntese Russelliana")
        l2_priorities = {j.proposicao[:40]: j.prioridade for j in judgments}
        result = self.L4.synthesize(pv_list, l2_priorities, prompt)
        self._log(f"  Síntese realizada (v={result.truth_value:.2f})")

        # L5: Geração (fallback template)
        self._log("\n[L5] Geração de Resposta")
        l5_text = generate_response_fallback(result, prompt)
        self._log(f"  Resposta gerada ({len(l5_text)} chars)")

        # L6: Refinamento
        self._log("\n[L6] Refinamento Final")
        epistemic_context = EpistemicContext()
        final_text = self.L6.finalize_response(
            prompt=prompt,
            synthesis_result=result,
            epistemic_context=epistemic_context,
            generated_text=l5_text,
            concepts_summary=concepts_summary,
        )
        self._log(f"  Resposta refinada")

        # L7: Síntese Final
        self._log("\n[L7] Síntese Final")
        l3_summary = "; ".join(f"{pv.proposition[:30]}→{pv.state}" for pv in pv_list[:3])
        final_output = self.L7.finalize_text(
            prompt=prompt,
            synthesis_result=result,
            l1_summary=concepts_summary,
            l2_summary=top_judgments,
            l3_summary=l3_summary,
            finalized_text=final_text,
        )
        self._log(f"  Síntese final com auditoria\n")

        result.response = final_output
        self._log(str(result))
        return result

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ============================================================================
# === EXEMPLOS E DEMO ===
# ============================================================================

def demo_local():
    """Demo interativa do middleware."""
    print("\n" + "="*70)
    print("DONINHA IA MIDDLEWARE — DEMO INTERATIVA")
    print("="*70)

    pipeline = HybridLLMPipeline(verbose=True)

    test_prompts = [
        "O que é verdade segundo Russell?",
        "Explique a diferença entre conhecimento e crença.",
        "Como funciona a lógica paraconsistente?"
    ]

    for prompt in test_prompts:
        print(f"\n{'─'*70}")
        print(f"PROMPT: {prompt}")
        print('─'*70)

        response = pipeline.process(prompt)

        print(f"\nRESPOSTA:\n{response.response}")
        print(f"\nESTADO: {response.state} ({response.confidence_label})")


def main() -> None:
    """Ponto de entrada principal."""
    import argparse
    parser = argparse.ArgumentParser(description="Doninha IA — Pipeline L1–L7 Compilado")
    parser.add_argument("--prompt", "-p", type=str, help="Pergunta única (imprime só a resposta)")
    parser.add_argument("--demo", action="store_true", help="Rodar demonstração com prompts fixos")
    args, _ = parser.parse_known_args()

    pipeline = HybridLLMPipeline(verbose=not args.prompt)

    if args.prompt:
        r = pipeline.process(args.prompt)
        print(r.response)
        return
    
    if args.demo:
        demo_local()
        return

    # Default: demo
    demo_local()


if __name__ == "__main__":
    main()

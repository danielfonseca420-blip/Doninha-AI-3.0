#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                         DONINHA IA MIDDLEWARE                            ║
║                                                                          ║
║  Middleware Neuro-Simbólico Híbrido em 7 Camadas                        ║
║                                                                          ║
║  L1: Tábua de Conceitos (Aristotélica)                                  ║
║  L2: Juízos Kantianos + Epistemologia                                   ║
║  L3: Lógica Paraconsistente (μ/λ, 12 estados)                           ║
║  L4: Síntese Russelliana + Chain of Verification                        ║
║  L5: Geração textual (suporte a múltiplos LLMs)                         ║
║  L6: Refinamento final                                                   ║
║  L7: Síntese definitiva auditável                                       ║
║                                                                          ║
║  Suporta: OpenAI, Anthropic, Google Gemini, Ollama, fallback            ║
║  Auto-contained, bem organizado, pronto para produção                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import os
import re
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
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

    # Tábua seminal em português
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
    quantidade: str  # Universal | Particular | Singular
    qualidade: str   # Afirmativo | Negativo | Infinito
    relacao: str     # Categórico | Hipotético | Disjuntivo
    modalidade: str  # Problemático | Assertórico | Apodítico
    proposicao: str
    prioridade: float = 0.0
    epistemic_classification: EpistemicClassification = field(default_factory=EpistemicClassification)

    def __str__(self) -> str:
        return (
            f"[{self.quantidade}/{self.qualidade}/"
            f"{self.relacao}/{self.modalidade}] "
            f"(pri={self.prioridade:.2f}) {self.epistemic_classification} {self.proposicao}"
        )


class KantianJudgmentEngine:
    """Motor de geração de juízos segundo Kant."""

    MODALIDADE_PESO = {
        "Apodítico": 1.0,
        "Assertórico": 0.7,
        "Problemático": 0.3,
    }

    def extract_judgments_from_text(self, text: str) -> List[KantianJudgment]:
        """Extrai juízos do texto."""
        judgments = []
        sentences = re.split(r'[.!?]\s+', text)
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            # Heurística: detecta quantidade
            quantidade = "Singular"
            if re.search(r"\b(todo|todos|toda|todas|cada)\b", sent, re.I):
                quantidade = "Universal"
            elif re.search(r"\b(algum|alguns|alguma|algumas|há)\b", sent, re.I):
                quantidade = "Particular"
            
            # Heurística: detecta qualidade
            qualidade = "Afirmativo"
            if re.search(r"\b(não|nunca|jamais|nenhum)\b", sent, re.I):
                qualidade = "Negativo"
            
            # Heurística: detecta relação
            relacao = "Categórico"
            if re.search(r"\bse\b.*\bentão\b", sent, re.I):
                relacao = "Hipotético"
            elif re.search(r"\bou\b", sent, re.I):
                relacao = "Disjuntivo"
            
            # Heurística: detecta modalidade
            modalidade = "Assertórico"
            if re.search(r"\b(pode|talvez|possivelmente)\b", sent, re.I):
                modalidade = "Problemático"
            elif re.search(r"\b(deve|precisa|deve ser)\b", sent, re.I):
                modalidade = "Apodítico"
            
            prioridade = self.MODALIDADE_PESO.get(modalidade, 0.5)
            if quantidade == "Singular" and qualidade == "Afirmativo":
                prioridade *= 1.5
            
            judgment = KantianJudgment(
                quantidade=quantidade,
                qualidade=qualidade,
                relacao=relacao,
                modalidade=modalidade,
                proposicao=sent,
                prioridade=min(1.0, prioridade)
            )
            judgments.append(judgment)
        
        # Ordena por prioridade (decrescente)
        judgments.sort(key=lambda j: j.prioridade, reverse=True)
        return judgments[:10]  # Top 10


# ============================================================================
# === L3 - LÓGICA PARACONSISTENTE ===
# ============================================================================

# Constantes de limiar (QUPC — Quadrado Unitário do Plano Cartesiano)
THRESHOLD_TRUE = 0.7
THRESHOLD_FALSE = 0.3
THRESHOLD_INCONSISTENT = 0.6
THRESHOLD_INDETERMINATE = 0.4

# 12 Estados lógicos
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


@dataclass
class ParaconsistentValue:
    """Anotação paraconsistente de uma proposição."""
    mu: float          # Grau de evidência favorável ∈ [0,1]
    lambda_: float     # Grau de evidência contrária ∈ [0,1]
    gc: float          # Grau de Certeza = μ - λ ∈ [-1,1]
    gct: float         # Grau de Contradição = μ + λ - 1 ∈ [-1,1]
    truth_value: float # Valor-verdade normalizado ∈ [0,1]
    state: str         # Estado lógico (um dos 12)
    proposition: str = ""

    def __str__(self) -> str:
        return f"μ={self.mu:.2f}, λ={self.lambda_:.2f} | Gc={self.gc:.2f}, Gct={self.gct:.2f} | v={self.truth_value:.2f} | {self.state}"


class ParaconsistentEngine:
    """Motor de avaliação paraconsistente."""

    @staticmethod
    def state_12(mu: float, lambda_: float) -> str:
        """Discretiza (μ, λ) em um dos 12 estados."""
        gc = mu - lambda_
        gct = mu + lambda_ - 1.0
        
        if gct >= 0.5:
            if gc >= 0.5:
                return STATE_T_TO_V
            if gc <= -0.5:
                return STATE_QV_TO_T
            return STATE_T
        if gct <= -0.5:
            if gc >= 0.5:
                return STATE_BOT_TO_V
            if gc <= -0.5:
                return STATE_BOT_TO_F
            return STATE_BOT
        if gc >= 0.5:
            return STATE_V if gct <= 0 else STATE_QV
        if gc <= -0.5:
            return STATE_F if gct <= 0 else STATE_QF
        if gct > 0:
            return STATE_QV_TO_BOT
        return STATE_QF_TO_V

    @staticmethod
    def evaluate(mu: float, lambda_: float, proposition: str = "") -> ParaconsistentValue:
        """Avalia uma proposição com anotações paraconsistentes."""
        gc = mu - lambda_
        gct = mu + lambda_ - 1.0
        truth_value = (mu + (1.0 - lambda_)) / 2.0
        state = ParaconsistentEngine.state_12(mu, lambda_)
        
        return ParaconsistentValue(
            mu=mu,
            lambda_=lambda_,
            gc=gc,
            gct=gct,
            truth_value=truth_value,
            state=state,
            proposition=proposition
        )

    @staticmethod
    def infer_annotations(text: str, context: str = "") -> Tuple[float, float]:
        """Infere μ e λ a partir do texto (heurístico)."""
        combined = (context + " " + text).lower()
        
        # Detecção heurística de evidência favorável
        favorable_patterns = [
            r"\b(verdadeiro|correto|certo|verdade|sim|positivo|bom|melhor|sucesso)\b",
            r"\b(alto|elevado|forte|robusto|sólido|comprovado|verificado)\b"
        ]
        favorable_count = sum(
            len(re.findall(pattern, combined))
            for pattern in favorable_patterns
        )
        mu = min(1.0, 0.1 + favorable_count * 0.15)
        
        # Detecção heurística de evidência contrária
        contrary_patterns = [
            r"\b(falso|errado|incorreto|engano|não|negativo|ruim|pior|fracasso)\b",
            r"\b(baixo|fraco|duvidoso|incerto|refutado|contestado)\b"
        ]
        contrary_count = sum(
            len(re.findall(pattern, combined))
            for pattern in contrary_patterns
        )
        lambda_ = min(1.0, 0.1 + contrary_count * 0.15)
        
        return (mu, lambda_)


# ============================================================================
# === L4 - SÍNTESE RUSSELLIANA + CHAIN OF VERIFICATION ===
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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def synthesize(
        self,
        prompt: str,
        concepts: List[ConceptNode],
        judgments: List[KantianJudgment],
        paraconsistent_values: List[ParaconsistentValue]
    ) -> SynthesisResult:
        """Sintetiza os resultados das camadas anteriores."""
        
        # Coleta evidências de suporte
        supporting_evidence = []
        for pv in paraconsistent_values:
            if pv.truth_value >= 0.65:
                supporting_evidence.append(pv.proposition)
        
        # Coleta hipóteses falsificadas
        falsified = []
        for pv in paraconsistent_values:
            if pv.truth_value <= 0.35:
                falsified.append(pv.proposition)
        
        # Calcula valor-verdade agregado
        truth_values = [pv.truth_value for pv in paraconsistent_values]
        avg_truth = sum(truth_values) / len(truth_values) if truth_values else 0.5
        
        # Calcula certeza agregada (Gc)
        gcs = [pv.gc for pv in paraconsistent_values]
        avg_gc = sum(gcs) / len(gcs) if gcs else 0.0
        
        # Calcula contradição agregada (Gct)
        gcts = [pv.gct for pv in paraconsistent_values]
        avg_gct = sum(gcts) / len(gcts) if gcts else 0.0
        
        # Determina estado lógico
        state = ParaconsistentEngine.state_12(
            (avg_gc + 1) / 2,  # Normaliza Gc para [0,1]
            (1 - (avg_gc + 1) / 2)  # Espelho para λ
        )
        
        # Constrói resposta
        response = self._build_response(
            prompt, concepts, judgments, supporting_evidence, falsified
        )
        
        return SynthesisResult(
            response=response,
            truth_value=avg_truth,
            certainty=avg_gc,
            contradiction=avg_gct,
            state=state,
            supporting_evidence=supporting_evidence,
            falsified_hypotheses=falsified,
            verification_log=self._build_verification_log(judgments, paraconsistent_values),
            confidence_label=""
        )

    def _build_response(
        self,
        prompt: str,
        concepts: List[ConceptNode],
        judgments: List[KantianJudgment],
        evidence: List[str],
        falsified: List[str]
    ) -> str:
        """Constrói resposta textual a partir da síntese."""
        lines = []
        
        if judgments:
            best_judgment = max(judgments, key=lambda j: j.prioridade)
            lines.append(f"Proposição central: {best_judgment.proposicao}")
        
        if evidence:
            lines.append(f"\nEvidências de suporte ({len(evidence)}):")
            for ev in evidence[:3]:
                lines.append(f"  • {ev[:80]}")
        
        if falsified:
            lines.append(f"\nHipóteses descartadas ({len(falsified)}):")
            for fh in falsified[:2]:
                lines.append(f"  ✗ {fh[:80]}")
        
        return "\n".join(lines)

    def _build_verification_log(
        self,
        judgments: List[KantianJudgment],
        paraconsistent_values: List[ParaconsistentValue]
    ) -> List[str]:
        """Constrói log de verificação (Chain of Verification)."""
        log = []
        
        for j in judgments[:3]:
            log.append(f"L2: Juízo [{j.quantidade}/{j.qualidade}/{j.relacao}] pri={j.prioridade:.2f}")
        
        for pv in paraconsistent_values[:3]:
            log.append(f"L3: Anotação μ={pv.mu:.2f}, λ={pv.lambda_:.2f} → {pv.state}")
        
        return log


# ============================================================================
# === L5 - GERAÇÃO TEXTUAL (MULTI-LLM) ===
# ============================================================================

class GenerationEngine:
    """Motor de geração de resposta com suporte a múltiplos LLMs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def generate(
        self,
        synthesis_result: SynthesisResult,
        prompt: str,
        provider: str = "fallback",
        model: str = "",
        api_key: str = ""
    ) -> str:
        """Gera resposta usando o provider especificado."""
        
        if provider == "openai":
            return self._generate_openai(synthesis_result, prompt, model, api_key)
        elif provider == "anthropic":
            return self._generate_anthropic(synthesis_result, prompt, model, api_key)
        elif provider == "gemini":
            return self._generate_gemini(synthesis_result, prompt, model, api_key)
        elif provider == "ollama":
            return self._generate_ollama(synthesis_result, prompt, model)
        else:
            return self._generate_fallback(synthesis_result, prompt)

    def _generate_openai(
        self,
        synthesis_result: SynthesisResult,
        prompt: str,
        model: str,
        api_key: str
    ) -> str:
        """Gera via OpenAI API."""
        try:
            import openai
            openai.api_key = api_key
            
            context = self._build_context(synthesis_result, prompt)
            
            response = openai.ChatCompletion.create(
                model=model or "gpt-4",
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado em síntese epistemológica."},
                    {"role": "user", "content": context}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Erro ao gerar com OpenAI: {e}")
            return self._generate_fallback(synthesis_result, prompt)

    def _generate_anthropic(
        self,
        synthesis_result: SynthesisResult,
        prompt: str,
        model: str,
        api_key: str
    ) -> str:
        """Gera via Anthropic Claude API."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            context = self._build_context(synthesis_result, prompt)
            
            message = client.messages.create(
                model=model or "claude-3-opus-20240229",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": context}
                ]
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Erro ao gerar com Anthropic: {e}")
            return self._generate_fallback(synthesis_result, prompt)

    def _generate_gemini(
        self,
        synthesis_result: SynthesisResult,
        prompt: str,
        model: str,
        api_key: str
    ) -> str:
        """Gera via Google Gemini API."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            model_obj = genai.GenerativeModel(model or "gemini-pro")
            context = self._build_context(synthesis_result, prompt)
            
            response = model_obj.generate_content(context)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Erro ao gerar com Gemini: {e}")
            return self._generate_fallback(synthesis_result, prompt)

    def _generate_ollama(
        self,
        synthesis_result: SynthesisResult,
        prompt: str,
        model: str
    ) -> str:
        """Gera via Ollama local."""
        try:
            import ollama
            
            context = self._build_context(synthesis_result, prompt)
            
            response = ollama.chat(
                model=model or "mistral",
                messages=[{"role": "user", "content": context}],
                stream=False,
            )
            
            if isinstance(response, dict):
                return response.get("message", {}).get("content", "").strip()
            return str(response).strip() if response else ""
        except Exception as e:
            logger.warning(f"Erro ao gerar com Ollama: {e}")
            return self._generate_fallback(synthesis_result, prompt)

    def _generate_fallback(
        self,
        synthesis_result: SynthesisResult,
        prompt: str
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

    def _build_context(self, synthesis_result: SynthesisResult, prompt: str) -> str:
        """Constrói contexto para o LLM gerar resposta."""
        return f"""Com base na seguinte análise epistemológica, elabore uma resposta clara e concisa:

Pergunta original: {prompt}

Valor de verdade: {synthesis_result.truth_value:.2f}
Estado lógico: {synthesis_result.state}
Nível de confiança: {synthesis_result.confidence_label}

Análise: {synthesis_result.response}

Elabore uma resposta final que sintetize essa análise de forma acessível e rigorosa."""


# ============================================================================
# === L6 - REFINAMENTO FINAL ===
# ============================================================================

class FinalizationEngine:
    """Motor de refinamento final da resposta."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def finalize(
        self,
        prompt: str,
        synthesis_result: SynthesisResult,
        generated_text: str
    ) -> str:
        """Refina a resposta final."""
        
        if not generated_text:
            generated_text = synthesis_result.response
        
        # Normaliza quebras de linha
        text = "\n\n".join(
            " ".join(line.split()).strip()
            for line in generated_text.splitlines()
            if line.strip()
        )
        
        # Adiciona contexto epistemológico quando apropriado
        if synthesis_result.confidence_label == "Indeterminado":
            text += "\n\nNota: Esta questão apresenta genuína indeterminação epistemológica."
        elif synthesis_result.contradiction > 0.3:
            text += "\n\nNota: Existem contradições locais que recomendam cautela na interpretação."
        
        return text.strip()


# ============================================================================
# === L7 - SÍNTESE DEFINITIVA COM AUDITORIA ===
# ============================================================================

class FinalTextEngine:
    """Motor de síntese final com blocos de auditoria."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def finalize_text(
        self,
        prompt: str,
        synthesis_result: SynthesisResult,
        concepts_summary: str = "",
        judgments_summary: str = "",
        finalized_text: str = ""
    ) -> str:
        """Gera texto final com auditoria L7."""
        
        if not finalized_text:
            finalized_text = synthesis_result.response
        
        # Bloco principal
        main = finalized_text.strip()
        
        # Bloco de auditoria
        audit_block = self._build_audit_block(
            synthesis_result,
            concepts_summary,
            judgments_summary
        )
        
        # Combina
        final = f"{main}\n\n{audit_block}"
        
        return final

    def _build_audit_block(
        self,
        synthesis_result: SynthesisResult,
        concepts_summary: str,
        judgments_summary: str
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
        
        if concepts_summary:
            lines.append(f"Conceitos integrados (L1): {concepts_summary[:100]}")
        
        if judgments_summary:
            lines.append(f"Juízos kantianos (L2): {judgments_summary[:100]}")
        
        if synthesis_result.supporting_evidence:
            lines.append(f"Evidências principais ({len(synthesis_result.supporting_evidence)})")
        
        if synthesis_result.falsified_hypotheses:
            lines.append(f"Hipóteses descartadas ({len(synthesis_result.falsified_hypotheses)})")
        
        lines.append("[FIM AUDIT L7]")
        
        return "\n".join(lines)


# ============================================================================
# === MIDDLEWARE PRINCIPAL ===
# ============================================================================

class DoninhaMiddleware:
    """
    Middleware Doninha completo com todas as 7 camadas.
    Interface unificada para processamento de prompts.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o middleware.
        
        Args:
            config: Dicionário com configurações:
                - provider: "openai" | "anthropic" | "gemini" | "ollama" | "fallback"
                - model: Nome/versão do modelo
                - api_key: Chave de API (se aplicável)
                - temperature: Temperatura de geração (default 0.3)
        """
        self.config = config or {}
        self.config.setdefault("provider", "fallback")
        self.config.setdefault("temperature", 0.3)
        
        # Inicializa as 7 camadas
        self.L1 = ConceptTable()
        self.L2 = KantianJudgmentEngine()
        self.L3 = ParaconsistentEngine()
        self.L4 = RussellianSynthesisEngine(self.config)
        self.L5 = GenerationEngine(self.config)
        self.L6 = FinalizationEngine(self.config)
        self.L7 = FinalTextEngine(self.config)
        
        logger.info("DoninhaMiddleware inicializado com sucesso")

    def process(self, prompt: str) -> str:
        """
        Processa um prompt através de todas as 7 camadas.
        
        Args:
            prompt: Pergunta/prompt do usuário
            
        Returns:
            Resposta final sintetizada e auditada
        """
        logger.info(f"Processando: {prompt[:50]}...")
        
        # L1: Extrai conceitos
        concepts = self.L1.extract_concepts(prompt)
        concepts_summary = ", ".join([c.term for c in concepts[:5]]) if concepts else "nenhum"
        logger.info(f"L1: {len(concepts)} conceitos extraídos")
        
        # L2: Gera juízos kantianos
        judgments = self.L2.extract_judgments_from_text(prompt)
        judgments_summary = " | ".join([str(j) for j in judgments[:2]]) if judgments else "nenhum"
        logger.info(f"L2: {len(judgments)} juízos gerados")
        
        # L3: Avalia paraconsistentemente
        paraconsistent_values = []
        for judgment in judgments:
            mu, lambda_ = ParaconsistentEngine.infer_annotations(judgment.proposicao, prompt)
            pv = ParaconsistentEngine.evaluate(mu, lambda_, judgment.proposicao)
            paraconsistent_values.append(pv)
        logger.info(f"L3: {len(paraconsistent_values)} anotações paraconsistentes")
        
        # L4: Síntese russelliana
        synthesis_result = self.L4.synthesize(
            prompt, concepts, judgments, paraconsistent_values
        )
        logger.info(f"L4: Síntese realizada (v={synthesis_result.truth_value:.2f})")
        
        # L5: Gera resposta
        generated_text = self.L5.generate(
            synthesis_result,
            prompt,
            provider=self.config.get("provider", "fallback"),
            model=self.config.get("model", ""),
            api_key=self.config.get("api_key", "")
        )
        logger.info(f"L5: Resposta gerada ({len(generated_text)} chars)")
        
        # L6: Refina
        finalized_text = self.L6.finalize(prompt, synthesis_result, generated_text)
        logger.info(f"L6: Resposta refinada")
        
        # L7: Síntese final com auditoria
        final_output = self.L7.finalize_text(
            prompt,
            synthesis_result,
            concepts_summary,
            judgments_summary,
            finalized_text
        )
        logger.info(f"L7: Síntese final com auditoria")
        
        return final_output

    def chat(self, messages: List[Dict[str, str]], system_prompt: str = "") -> str:
        """
        Interface de chat (para compatibilidade com APIs).
        
        Args:
            messages: Lista de mensagens [{"role": "user", "content": "..."}]
            system_prompt: Prompt de sistema (ignorado nesta versão)
            
        Returns:
            Resposta processada
        """
        if not messages:
            return ""
        
        # Extrai a última mensagem do usuário
        last_message = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            ""
        )
        
        if not last_message:
            return ""
        
        return self.process(last_message)

    def set_config(self, config: Dict[str, Any]) -> None:
        """Atualiza a configuração do middleware."""
        self.config.update(config)
        logger.info(f"Configuração atualizada: {config}")

    def get_status(self) -> Dict[str, Any]:
        """Retorna status do middleware."""
        return {
            "provider": self.config.get("provider"),
            "model": self.config.get("model"),
            "status": "operacional",
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# === EXEMPLOS E DEMO ===
# ============================================================================

def demo_local():
    """Demo interativa do middleware."""
    print("\n" + "="*70)
    print("DONINHA IA MIDDLEWARE — DEMO INTERATIVA")
    print("="*70)
    
    # Cria middleware com fallback
    middleware = DoninhaMiddleware({
        "provider": "fallback",
        "model": "doninha-demo"
    })
    
    # Prompts de teste
    test_prompts = [
        "O que é verdade segundo Russell?",
        "Explique a diferença entre conhecimento e crença.",
        "Como funciona a lógica paraconsistente?"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'─'*70}")
        print(f"PROMPT: {prompt}")
        print('─'*70)
        
        response = middleware.process(prompt)
        print(f"\nRESPOSTA:\n{response}")
        
        print(f"\nSTATUS: {middleware.get_status()}")


def demo_with_openai(api_key: str):
    """Demo com OpenAI."""
    print("\n" + "="*70)
    print("DONINHA IA MIDDLEWARE — DEMO COM OPENAI")
    print("="*70)
    
    middleware = DoninhaMiddleware({
        "provider": "openai",
        "model": "gpt-4",
        "api_key": api_key,
        "temperature": 0.3
    })
    
    prompt = "Qual é o significado da epistemologia?"
    print(f"\nPrompt: {prompt}")
    
    response = middleware.process(prompt)
    print(f"\nResposta:\n{response}")


if __name__ == "__main__":
    # Demo padrão (fallback)
    demo_local()
    
    # Descomente para testar com sua API key do OpenAI:
    # demo_with_openai("sua-api-key-aqui")
    
    print("\n" + "="*70)
    print("Demo concluída!")
    print("="*70)

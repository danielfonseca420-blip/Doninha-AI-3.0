"""
CAMADA L2 — Tábua de Juízos Kantianos
======================================
Antes de qualquer cálculo estatístico o prompt é destrinchado nas
doze categorias da Tábua dos Juízos (Kritik der reinen Vernunft, §9).

Dimensões:
  Quantidade  → Universal | Particular | Singular
  Qualidade   → Afirmativo | Negativo | Infinito
  Relação     → Categórico | Hipotético | Disjuntivo
  Modalidade  → Problemático | Assertórico | Apodítico

Cada hipótese gerada recebe um peso de prioridade; o Juízo Singular
Afirmativo Assertórico tem prioridade máxima (é a resposta-alvo).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from l1_concept_table import ConceptNode, ConceptTable
import re

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


# ─────────────────────────────────────────────────────────────────────────────
# Estruturas de dados
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpistemicClassification:
    """Classificação epistemológica sem restrição T+I+F=1."""
    truth: float = 0.0           # T ∈ [0,1] — grau de verdade
    indeterminacy: float = 0.0   # I ∈ [0,1] — grau de indeterminação
    falsity: float = 0.0         # F ∈ [0,1] — grau de falsidade
    classification: str = "indeterminado"  # paraconsistência | incompletude | vagueza | assertiva_confiante | indeterminado

    def __post_init__(self):
        self.classification = self._classify()

    def _classify(self) -> str:
        """Aplica regras epistemológicas para classificar."""
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
    """Uma proposição refinada segundo a tábua dos juízos."""
    quantidade:  str   # Universal | Particular | Singular
    qualidade:   str   # Afirmativo | Negativo | Infinito
    relacao:     str   # Categórico | Hipotético | Disjuntivo
    modalidade:  str   # Problemático | Assertórico | Apodítico
    proposicao:  str   # texto da hipótese
    prioridade:  float = 0.0  # 0.0 → 1.0  (1.0 = resposta-alvo)
    epistemic_classification: EpistemicClassification = field(default_factory=EpistemicClassification)

    def __str__(self) -> str:
        return (
            f"[{self.quantidade}/{self.qualidade}/"
            f"{self.relacao}/{self.modalidade}] "
            f"(pri={self.prioridade:.2f}) {self.epistemic_classification} {self.proposicao}"
        )


@dataclass
class SyntaxProfile:
    """
    Perfil sintático mínimo extraído do enunciado segundo a gramática
    (aproximação heurística baseada em listas inspiradas em grammar.txt).
    """
    quantifier_subject: Optional[str] = None   # "all", "some", "this", etc.
    quantifier_predicate: Optional[str] = None
    has_negation: bool = False
    has_infinite_like: bool = False           # construções do tipo "not-X"
    is_conditional: bool = False              # presença de "if", "then"
    is_disjunctive: bool = False              # presença de "or"
    modality_markers: Tuple[str, ...] = ()    # "can", "must", "might", etc.


# ─────────────────────────────────────────────────────────────────────────────
# Regras de prioridade entre modalidades (herança da "parte fraca")
# ─────────────────────────────────────────────────────────────────────────────
MODALIDADE_PESO = {
    "Apodítico":    1.0,
    "Assertórico":  0.7,
    "Problemático": 0.4,
}
QUANTIDADE_PESO = {
    "Singular":   1.0,
    "Particular": 0.6,
    "Universal":  0.3,
}
QUALIDADE_PESO = {
    "Afirmativo": 1.0,
    "Infinito":   0.6,
    "Negativo":   0.4,
}
RELACAO_PESO = {
    "Categórico":  1.0,
    "Hipotético":  0.7,
    "Disjuntivo":  0.5,
}


def _priority(j: KantianJudgment) -> float:
    return (
        MODALIDADE_PESO[j.modalidade]
        * QUANTIDADE_PESO[j.quantidade]
        * QUALIDADE_PESO[j.qualidade]
        * RELACAO_PESO[j.relacao]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Motor de geração de juízos
# ─────────────────────────────────────────────────────────────────────────────

class BERTAssertionClassifier:
    """Classificador baseado em BERT para juízos assertóricos.

    Processa proposições e retorna (T, I, F) sem restrição T+I+F=1,
    capturando paraconsistência, incompletude e vagueza.
    """

    DOMAIN_CANDIDATES = {
        "físico": [
            "empiricamente verificado",
            "teoricamente plausível",
            "logicamente contraditório",
            "empiricamente indeterminado",
        ],
        "lógica": [
            "logicamente contraditório",
            "teoricamente plausível",
            "empiricamente indeterminado",
            "empiricamente verificado",
        ],
        "cognitivo": [
            "empiricamente verificado",
            "teoricamente plausível",
            "empiricamente indeterminado",
            "logicamente contraditório",
        ],
        "filosófico": [
            "teoricamente plausível",
            "empiricamente indeterminado",
            "logicamente contraditório",
            "empiricamente verificado",
        ],
        "geral": [
            "teoricamente plausível",
            "empiricamente verificado",
            "logicamente contraditório",
            "empiricamente indeterminado",
        ],
    }
    GENERIC_CANDIDATES = [
        "empiricamente verificado",
        "teoricamente plausível",
        "logicamente contraditório",
        "empiricamente indeterminado",
    ]

    def __init__(self):
        self.classifier = None
        if pipeline is not None:
            try:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="bert-base-multilingual-uncased",
                )
            except Exception:
                pass

    def classify(self, proposition: str, domain: str = "geral") -> EpistemicClassification:
        """Classifica uma proposição em (T, I, F) usando candidatos por domínio."""
        if self.classifier is None:
            return self._heuristic_classify(proposition)

        candidates = self.DOMAIN_CANDIDATES.get(domain, self.GENERIC_CANDIDATES)
        try:
            result = self.classifier(proposition, candidates, multi_class=True)
            scores = {label: score for label, score in zip(result["labels"], result["scores"])}
            return EpistemicClassification(
                truth=scores.get("empiricamente verificado", 0.0)
                + scores.get("teoricamente plausível", 0.0) * 0.75,
                indeterminacy=scores.get("empiricamente indeterminado", 0.0),
                falsity=scores.get("logicamente contraditório", 0.0),
            )
        except Exception:
            return self._heuristic_classify(proposition)

    def _heuristic_classify(self, proposition: str) -> EpistemicClassification:
        """Classificação heurística quando BERT não está disponível."""
        text = proposition.lower()
        t, i, f = 0.5, 0.3, 0.2

        if "verdadeiro" in text or "é" in text or "sempre" in text:
            t = 0.8
            i = 0.1
            f = 0.1
        elif "falso" in text or "nunca" in text or "não é" in text:
            t = 0.1
            i = 0.1
            f = 0.8
        elif "pode" in text or "talvez" in text or "possível" in text:
            t = 0.4
            i = 0.5
            f = 0.3
        elif "contraditório" in text or "e" in text and "ou" in text:
            t = 0.6
            i = 0.3
            f = 0.7
        elif "indeterminado" in text or "indefinido" in text:
            t = 0.3
            i = 0.7
            f = 0.3
        elif "incompleto" in text or "insuficiente" in text:
            t = 0.2
            i = 0.6
            f = 0.2

        return EpistemicClassification(truth=round(t, 3), indeterminacy=round(i, 3), falsity=round(f, 3))


class KantianJudgmentEngine:
    """
    Recebe um prompt e a lista de ConceptNodes extraídos por L1 e devolve
    as 12 hipóteses estruturadas segundo a tábua kantiana.
    
    Para juizos assertóricos, aplica classificação BERT com (T, I, F).
    """

    def __init__(self, concept_table: ConceptTable) -> None:
        self.ct = concept_table
        self.bert_classifier = BERTAssertionClassifier()

    # ------------------------------------------------------------------ #
    # API pública                                                          #
    # ------------------------------------------------------------------ #

    def refine(self, prompt: str, concepts: List[ConceptNode]) -> List[KantianJudgment]:
        """
        Gera as hipóteses kantianas para o prompt e as ordena por
        prioridade descendente.
        """
        subject, predicates = self._parse_prompt(prompt, concepts)
        syntax = self._analyze_syntax(prompt)
        judgments: List[KantianJudgment] = []

        domain = self._infer_domain(concepts)
        for pred in predicates:
            antonym = self._antonym_of(pred, concepts)
            hypernym = self._hypernym_of(pred, concepts)

            # ── Juízo principal guiado pela gramática ───────────────────
            qt = self._infer_quantity(syntax)
            ql = self._infer_quality(syntax)
            rel = self._infer_relation(syntax)
            mod = self._infer_modality(syntax)

            base_prop = f"{subject} é {pred}"
            if syntax.has_negation and antonym:
                base_prop = f"{subject} não é {antonym}"

            j = self._make(qt, ql, rel, mod, base_prop)
            if j.modalidade == "Assertórico":
                j.epistemic_classification = self.bert_classifier.classify(base_prop, domain=domain)
            judgments.append(j)

            # ── Variações canônicas (mantidas, mas ancoradas em L1) ─────
            judgments.append(self._make(
                "Universal", "Afirmativo", "Categórico", "Apodítico",
                f"Todo(a) {subject} com propriedade extrema é {pred}",
            ))
            judgments.append(self._make(
                "Particular", "Afirmativo", "Hipotético", "Problemático",
                f"Algum(a) {subject} pode ser {pred}",
            ))
            j1 = self._make(
                "Singular", "Afirmativo", "Categórico", "Assertórico",
                f"Este(a) {subject} específico é {pred}",
            )
            j1.epistemic_classification = self.bert_classifier.classify(j1.proposicao, domain=domain)
            judgments.append(j1)

            prop2 = (f"Este(a) {subject} não é {antonym}" if antonym else
                     f"Este(a) {subject} não possui a propriedade oposta a {pred}")
            j2 = self._make("Singular", "Negativo", "Categórico", "Assertórico", prop2)
            j2.epistemic_classification = self.bert_classifier.classify(j2.proposicao, domain=domain)
            judgments.append(j2)
            j3 = self._make(
                "Singular", "Infinito", "Categórico", "Assertórico",
                f"Este(a) {subject} é não-{antonym}" if antonym else
                f"Este(a) {subject} é indeterminado em relação a {pred}",
            )
            j3.epistemic_classification = self.bert_classifier.classify(j3.proposicao, domain=domain)
            judgments.append(j3)

            judgments.append(self._make(
                "Universal", "Afirmativo", "Hipotético", "Apodítico",
                f"Se {subject} possui condição X, então é {pred}",
            ))
            j4 = self._make(
                "Universal", "Afirmativo", "Disjuntivo", "Assertórico",
                f"{subject} é {pred} OU {antonym} OU intermediário"
                if antonym else f"{subject} é {pred} ou outra propriedade",
            )
            j4.epistemic_classification = self.bert_classifier.classify(j4.proposicao, domain=domain)
            judgments.append(j4)

            judgments.append(self._make(
                "Singular", "Afirmativo", "Categórico", "Problemático",
                f"Este(a) {subject} pode ser {pred}?",
            ))
            j5 = self._make(
                "Singular", "Afirmativo", "Hipotético", "Assertórico",
                f"Este(a) {subject} é {pred} em razão das condições observadas",
            )
            j5.epistemic_classification = self.bert_classifier.classify(j5.proposicao, domain=domain)
            judgments.append(j5)
            judgments.append(self._make(
                "Universal", "Afirmativo", "Categórico", "Apodítico",
                f"{subject} deve ser {pred} quando condições necessárias presentes",
            ))

            # ── HIPÓTESES COM INTERMEDIÁRIOS (hiperonímia) ───────────────
            if hypernym:
                j6 = self._make(
                    "Singular", "Afirmativo", "Categórico", "Assertórico",
                    f"Este(a) {subject} pertence à categoria {hypernym}",
                )
                j6.epistemic_classification = self.bert_classifier.classify(j6.proposicao, domain=domain)
                judgments.append(j6)
            if antonym:
                j7 = self._make(
                    "Singular", "Negativo", "Disjuntivo", "Assertórico",
                    f"Este(a) {subject} não é {pred} nem {antonym}: "
                    f"admite valor intermediário",
                )
                j7.epistemic_classification = self.bert_classifier.classify(j7.proposicao, domain=domain)
                judgments.append(j7)

        # Calcula prioridades e ordena
        for j in judgments:
            j.prioridade = _priority(j)
        judgments.sort(key=lambda j: j.prioridade, reverse=True)
        return judgments

    def _infer_domain(self, concepts: List[ConceptNode]) -> str:
        """Inferência simples de domínio majoritário a partir dos conceitos extraídos."""
        if not concepts:
            return "geral"
        domain_counts = {}
        for concept in concepts:
            domain = concept.domain.lower().strip() if concept.domain else "geral"
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return max(domain_counts, key=domain_counts.get)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make(qt, ql, rel, mod, prop) -> KantianJudgment:
        j = KantianJudgment(
            quantidade=qt, qualidade=ql, relacao=rel,
            modalidade=mod, proposicao=prop,
        )
        j.prioridade = _priority(j)
        return j

    @staticmethod
    def _parse_prompt(prompt: str, concepts: List[ConceptNode]) -> Tuple[str, List[str]]:
        """
        Extrai sujeito e predicados candidatos do prompt de forma simples.
        Em produção seria substituído por um parser sintático.
        """
        tokens = re.findall(r"[a-záàãâéêíóôõúüçA-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]+", prompt.lower())
        known = {c.term.lower() for c in concepts}
        subject = tokens[0] if tokens else "entidade"
        predicates = [t for t in tokens[1:] if t in known] or ["indeterminado"]
        return subject, predicates

    # ------------------------------------------------------------------ #
    # Análise sintática inspirada em grammar.txt                         #
    # ------------------------------------------------------------------ #

    def _analyze_syntax(self, prompt: str) -> SyntaxProfile:
        """
        Extrai um perfil sintático mínimo usando listas de palavras
        alinhadas aos capítulos de determiners, modals, negatives e
        conjunctions da grammar COBUILD.
        """
        text = prompt.lower()
        tokens = re.findall(r"[a-záàãâéêíóôõúüç]+", text)

        quant_all = {"all", "every", "each"}
        quant_some = {"some", "many", "several", "few", "a few"}
        quant_singular = {"this", "that", "these", "those", "a", "an", "one"}

        neg_markers = {"not", "no", "never", "none", "nothing", "nowhere"}
        infinite_patterns = {"not-", "non-"}

        cond_markers = {"if", "provided", "unless", "whenever", "as long as"}
        disj_markers = {"or", "either"}

        modal_poss = {"can", "could", "may", "might"}
        modal_necess = {"must", "have to", "need to", "should", "ought"}

        has_neg = any(tok in neg_markers for tok in tokens)
        has_inf = any(pat in text for pat in infinite_patterns)
        is_cond = any(tok in cond_markers for tok in tokens)
        is_disj = any(tok in disj_markers for tok in tokens)

        mods: list[str] = []
        for tok in tokens:
            if tok in modal_poss or tok in modal_necess:
                mods.append(tok)

        q_subj: Optional[str] = None
        q_pred: Optional[str] = None

        if tokens:
            first = tokens[0]
            if first in quant_all:
                q_subj = "all"
            elif first in quant_some:
                q_subj = "some"
            elif first in quant_singular:
                q_subj = "this"

        return SyntaxProfile(
            quantifier_subject=q_subj,
            quantifier_predicate=q_pred,
            has_negation=has_neg,
            has_infinite_like=has_inf,
            is_conditional=is_cond,
            is_disjunctive=is_disj,
            modality_markers=tuple(mods),
        )

    def _infer_quantity(self, syntax: SyntaxProfile) -> str:
        if syntax.quantifier_subject == "all":
            return "Universal"
        if syntax.quantifier_subject == "some":
            return "Particular"
        if syntax.quantifier_subject == "this":
            return "Singular"
        return "Singular"

    def _infer_quality(self, syntax: SyntaxProfile) -> str:
        if syntax.has_infinite_like:
            return "Infinito"
        if syntax.has_negation:
            return "Negativo"
        return "Afirmativo"

    def _infer_relation(self, syntax: SyntaxProfile) -> str:
        if syntax.is_conditional:
            return "Hipotético"
        if syntax.is_disjunctive:
            return "Disjuntivo"
        return "Categórico"

    def _infer_modality(self, syntax: SyntaxProfile) -> str:
        markers = {m for m in syntax.modality_markers}
        if any(m in {"must", "have", "need", "should", "ought"} for m in markers):
            return "Apodítico"
        if any(m in {"can", "could", "may", "might"} for m in markers):
            return "Problemático"
        return "Assertórico"

    def _antonym_of(self, term: str, concepts: List[ConceptNode]) -> str:
        node = self.ct.get(term)
        if node and node.antonyms:
            return node.antonyms[0]
        return ""

    def _hypernym_of(self, term: str, concepts: List[ConceptNode]) -> str:
        node = self.ct.get(term)
        if node and node.hypernyms:
            return node.hypernyms[0]
        return ""

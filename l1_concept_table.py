"""
CAMADA L1 — Tábua de Conceitos (Aristóteles: Categorias)
=========================================================
Mapeia cada termo do prompt a relações semânticas fixas:
  - Sinonímia   : mesma denotação
  - Antonímia   : oposição semântica direta
  - Hiponímia   : relação específico → geral
  - Homonímia   : mesma forma, sentidos distintos
  - Paronímia   : semelhança formal, sentidos distintos

As relações são BINÁRIAS nesta camada — elimina a necessidade de
defuzzificação posterior na camada L3.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re
import json
import os
from knowledge_base import get_domain_knowledge_base


@dataclass
class ConceptNode:
    """Um conceito na tábua, com todas as suas relações."""
    term: str
    definition: str = ""
    synonyms:   List[str] = field(default_factory=list)
    antonyms:   List[str] = field(default_factory=list)
    hyponyms:   List[str] = field(default_factory=list)   # mais específicos
    hypernyms:  List[str] = field(default_factory=list)   # mais gerais
    homonyms:   Dict[str, str] = field(default_factory=dict)  # sentido → definição
    paronyms:   List[str] = field(default_factory=list)
    domain:     str = "geral"
    application_context: str = ""
    canonical_source: str = ""
    canonical_context: Dict[str, str] = field(default_factory=dict)  # Verificação de atribuição canônica


class ConceptTable:
    """
    Tábua de conceitos fixos.  Em produção seria alimentada por um
    dicionário / ontologia formal (WordNet-PT, OpenWordNet-PT, etc.).
    Aqui usamos um conjunto seminal suficiente para demonstrar todas
    as camadas do modelo.
    """

    def __init__(self) -> None:
        self._table: Dict[str, ConceptNode] = {}
        # Tábua seminal em português
        self._build_seed_table()
        # Banco de conceitos em inglês aprendido de dicionário externo (se existir)
        self._load_external_concepts()

    # ------------------------------------------------------------------ #
    # API pública                                                          #
    # ------------------------------------------------------------------ #

    def get(self, term: str) -> Optional[ConceptNode]:
        return self._table.get(self._normalize(term))

    def extract_concepts(self, text: str, llm_context: Optional[str] = None, domain: str = "geral", config: Optional[Dict] = None) -> List[ConceptNode]:
        """Extrai e retorna os nós de todos os termos encontrados no texto."""
        tokens = re.findall(r"[a-záàãâéêíóôõúüçA-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]+", text)
        seen, result = set(), []
        for tok in tokens:
            key = self._normalize(tok)
            if key not in seen:
                node = self._table.get(key)
                if node:
                    seen.add(key)
                    result.append(self._clone_node(node))

        if result:
            self._enrich_concepts_with_application_context(result, text, llm_context, domain, config)
            combined_text = f"{llm_context.strip()} {text}" if llm_context else text
            result = [
                node for node in result
                if LogicLMSymbolicSolver.is_context_compatible(node, combined_text)
            ]
        return result

    def add(self, node: ConceptNode) -> None:
        self._table[self._normalize(node.term)] = node

    def relation_type(self, term_a: str, term_b: str) -> str:
        """Retorna o tipo de relação semântica entre dois termos."""
        a = self._normalize(term_a)
        b = self._normalize(term_b)
        node_a = self._table.get(a)
        if not node_a:
            return "desconhecida"
        if b in [self._normalize(s) for s in node_a.synonyms]:
            return "sinonímia"
        if b in [self._normalize(s) for s in node_a.antonyms]:
            return "antonímia"
        if b in [self._normalize(s) for s in node_a.hyponyms]:
            return "hiponímia"
        if b in [self._normalize(s) for s in node_a.hypernyms]:
            return "hiperonímia"
        if b in [self._normalize(s) for s in node_a.paronyms]:
            return "paronímia"
        if b in [self._normalize(k) for k in node_a.homonyms]:
            return "homonímia"
        return "sem_relação_direta"

    # ------------------------------------------------------------------ #
    # Construção da tábua seminal                                          #
    # ------------------------------------------------------------------ #

    def _clone_node(self, node: ConceptNode) -> ConceptNode:
        return ConceptNode(
            term=node.term,
            definition=node.definition,
            synonyms=list(node.synonyms),
            antonyms=list(node.antonyms),
            hyponyms=list(node.hyponyms),
            hypernyms=list(node.hypernyms),
            homonyms=dict(node.homonyms),
            paronyms=list(node.paronyms),
            domain=node.domain,
            application_context="",
            canonical_source=node.canonical_source,
            canonical_context=dict(node.canonical_context),
        )

    def _enrich_concepts_with_application_context(
        self,
        concepts: List[ConceptNode],
        prompt: str,
        llm_context: Optional[str] = None,
        domain: str = "geral",
        config: Optional[Dict] = None,
    ) -> None:
        """Aplica o solver simbólico Logic-LM para adicionar contexto de uso aos conceitos."""
        LogicLMSymbolicSolver.enrich(concepts, prompt, llm_context, domain, config)

    def _build_seed_table(self) -> None:
        entries = [
            ConceptNode(
                term="quente",
                definition="Que possui temperatura alta.",
                synonyms=["aquecido", "cálido", "morno", "tépido"],
                antonyms=["frio", "gelado", "fresco"],
                hypernyms=["temperatura"],
                hyponyms=["escaldante", "ardente"],
                domain="físico",
                canonical_source="Newton - Philosophiae Naturalis Principia Mathematica - Livro I",
            ),
            ConceptNode(
                term="frio",
                definition="Que possui temperatura baixa.",
                synonyms=["gelado", "fresco", "frígido"],
                antonyms=["quente", "aquecido", "cálido"],
                hypernyms=["temperatura"],
                hyponyms=["congelado", "glacial"],
                domain="físico",
                canonical_source="Newton - Philosophiae Naturalis Principia Mathematica - Livro I",
            ),
            ConceptNode(
                term="morno",
                definition="Entre quente e frio; tépido.",
                synonyms=["tépido", "ameno"],
                antonyms=["escaldante", "glacial"],
                hypernyms=["temperatura", "quente", "frio"],
                hyponyms=[],
                domain="físico",
                canonical_source="Galen - De Temperamentis - Seção 3",
            ),
            ConceptNode(
                term="temperatura",
                definition="Grandeza física que mede o grau de calor de um corpo.",
                synonyms=["calor", "grau"],
                antonyms=[],
                hypernyms=["grandeza_física"],
                hyponyms=["quente", "frio", "morno"],
                domain="físico",
                canonical_source="Galileu - Discorsi e Dimostrazioni Matematiche - Seção 2",
            ),
            ConceptNode(
                term="água",
                definition="Substância H2O, geralmente em estado líquido.",
                synonyms=["H2O", "líquido"],
                antonyms=[],
                hypernyms=["substância", "fluido"],
                hyponyms=["vapor", "gelo"],
                domain="físico",
                canonical_source="Newton - Opticks - Definição 19",
            ),
            ConceptNode(
                term="verdadeiro",
                definition="Que está de acordo com os fatos ou a realidade.",
                synonyms=["correto", "real", "factual"],
                antonyms=["falso", "incorreto", "fictício"],
                hypernyms=["valor_lógico"],
                domain="lógica",
                canonical_source="Aristóteles - Metafísica - Livro Gamma",
                canonical_context={
                    "lógica_clássica": "Aristóteles - Metafísica: valor de verdade binário, NÃO lógica paraconsistente",
                    "epistemologia": "Platão - Teeteto: correspondência com realidade, NÃO coerência pura"
                }
            ),
            ConceptNode(
                term="falso",
                definition="Que não corresponde aos fatos ou à realidade.",
                synonyms=["incorreto", "errado", "fictício"],
                antonyms=["verdadeiro", "correto", "real"],
                hypernyms=["valor_lógico"],
                domain="lógica",
                canonical_source="Aristóteles - Metafísica - Livro Gamma",
                canonical_context={
                    "lógica_clássica": "Aristóteles - Metafísica: negação do verdadeiro, NÃO dialética hegeliana"
                }
            ),
            ConceptNode(
                term="banco",
                definition="Móvel para sentar; instituição financeira; repositório de dados.",
                synonyms=[],
                antonyms=[],
                hypernyms=[],
                homonyms={
                    "assento": "móvel para sentar",
                    "financeiro": "instituição financeira",
                    "dados": "repositório de dados",
                },
                domain="geral",
            ),
            ConceptNode(
                term="eminente",
                definition="Pessoa ilustre ou notável.",
                synonyms=["ilustre", "notável"],
                antonyms=[],
                paronyms=["iminente"],
                domain="geral",
            ),
            ConceptNode(
                term="iminente",
                definition="Que está prestes a acontecer.",
                synonyms=["próximo", "imediato"],
                antonyms=[],
                paronyms=["eminente"],
                domain="geral",
            ),
            ConceptNode(
                term="inteligência",
                definition="Capacidade de compreender, raciocinar e resolver problemas.",
                synonyms=["cognição", "raciocínio", "entendimento"],
                antonyms=["ignorância", "estupidez"],
                hypernyms=["capacidade_mental"],
                domain="cognitivo",
            ),
            ConceptNode(
                term="conhecimento",
                definition="Ato ou efeito de conhecer; saber, ciência, erudição.",
                synonyms=["saber", "ciência", "erudição"],
                antonyms=["ignorância", "desconhecimento"],
                hypernyms=["epistemologia"],
                domain="filosófico",
                canonical_context={
                    "epistemologia": "Platão - Teeteto: justificação verdadeira, NÃO opinião infundada",
                    "kantiano": "Kant - Crítica da Razão Pura: a priori vs a posteriori, NÃO empirismo puro"
                }
            ),
            ConceptNode(
                term="verdade",
                definition="Conformidade entre o que se diz e o que é.",
                synonyms=["veracidade", "factualidade", "realidade"],
                antonyms=["mentira", "falsidade", "ilusão"],
                hypernyms=["epistemologia"],
                domain="filosófico",
                canonical_context={
                    "platônico": "Platão - República: ideias eternas, NÃO relativismo",
                    "aristotélico": "Aristóteles - Metafísica: correspondência, NÃO coerência"
                }
            ),
            ConceptNode(
                term="síntese regulativa",
                definition="Princípio que orienta o conhecimento sem constituí-lo.",
                synonyms=["regulativo", "orientador"],
                antonyms=[],
                hypernyms=["epistemologia", "kantismo"],
                domain="filosófico",
                canonical_source="Kant - Crítica da Razão Pura",
                canonical_context={
                    "kantismo": "Kant, CRP: princípio regulativo do conhecimento, NÃO Russell"
                }
            ),
        ]
        for node in entries:
            self.add(node)

    @staticmethod
    def _normalize(term: str) -> str:
        return term.strip().lower()

    # ------------------------------------------------------------------ #
    # Carregamento de conceitos externos (ex.: dicionário em inglês)      #
    # ------------------------------------------------------------------ #

    def _load_external_concepts(self) -> None:
        """
        Carrega conceitos adicionais de um banco gerado a partir do
        dicionário em inglês (arquivo JSON se existir).

        Formato esperado (lista de objetos):
          {
            "term": "abacus",
            "definition": "Frame with beads for calculating...",
            "synonyms": [],
            "antonyms": [],
            "hyponyms": [],
            "hypernyms": [],
            "domain": "geral"
          }
        """
        base_dir = os.path.dirname(__file__) or "."
        json_path = os.path.join(base_dir, "data", "concepts_en.json")
        if not os.path.exists(json_path):
            return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            return

        for item in items:
            term = item.get("term")
            if not term:
                continue
            node = ConceptNode(
                term=term,
                definition=item.get("definition", ""),
                synonyms=item.get("synonyms", []),
                antonyms=item.get("antonyms", []),
                hyponyms=item.get("hyponyms", []),
                hypernyms=item.get("hypernyms", []),
                homonyms=item.get("homonyms", {}),
                paronyms=item.get("paronyms", []),
                domain=item.get("domain", "geral"),
                application_context="",
                canonical_source=item.get("canonical_source", ""),
            )
            key = self._normalize(term)
            if key not in self._table:
                self._table[key] = node


class LogicLMSymbolicSolver:
    """Processador simbólico inspirado em LLM-Symbolic Solver Logic-LM.

    Este módulo faz uma pesquisa contextual nos parâmetros de entrada da
    LLM base da IA Doninha e acrescenta à definição dos conceitos uma nota
    de aplicação prática para o prompt atual.
    """

    CONTEXTUAL_KEYWORDS = {
        "físico": ["temperatura", "calor", "energia", "massa", "volume"],
        "lógica": ["verdade", "falso", "proposição", "argumento", "inferencia", "inferência"],
        "cognitivo": ["raciocínio", "inteligência", "compreender", "resolver", "pensar"],
        "filosófico": ["verdade", "conhecimento", "epistemologia", "realidade", "ética"],
        "geral": ["aplicação", "uso", "contexto", "pergunta", "problema"],
    }

    @classmethod
    def enrich(
        cls,
        concepts: List[ConceptNode],
        prompt: str,
        llm_context: Optional[str] = None,
        domain: str = "geral",
        config: Optional[Dict] = None,
    ) -> List[ConceptNode]:
        text = prompt.strip()
        if llm_context:
            text = f"{llm_context.strip()} {text}"
        lower_text = text.lower()

        # Carrega KB específico do domínio
        kb = get_domain_knowledge_base(domain, config, query_for_rag=text)

        for node in concepts:
            node.application_context = cls._infer_application_context(node, lower_text, concepts, kb)
        return concepts

    @classmethod
    def _infer_application_context(
        cls,
        node: ConceptNode,
        text: str,
        concepts: List[ConceptNode],
        kb: Dict[str, float],
    ) -> str:
        base_context = ""
        if node.term.lower() in text:
            if not cls.is_context_compatible(node, text):
                return ""
            relation = cls._infer_relation(node, text, concepts)
            if relation:
                base_context = relation
        else:
            base_context = cls._default_context(node)

        # Enriquece com termos relevantes do KB do domínio
        relevant_terms = [term for term, score in kb.items() if term.lower() in text.lower() and score > 0.5]
        if relevant_terms:
            kb_context = f" Contexto de conhecimento: {', '.join(relevant_terms[:3])}."
            base_context += kb_context

        return base_context

    @classmethod
    def is_context_compatible(
        cls,
        node: ConceptNode,
        text: str,
    ) -> bool:
        if not node.canonical_source:
            return True
        lower_text = text.lower()
        if node.term.lower() in lower_text:
            return True
        if node.domain:
            domain_keywords = cls.CONTEXTUAL_KEYWORDS.get(node.domain, [])
            if any(keyword in lower_text for keyword in domain_keywords):
                return True
        canonical_keywords = cls._extract_source_keywords(node.canonical_source)
        if any(keyword in lower_text for keyword in canonical_keywords):
            return True
        if node.application_context and any(
            part in lower_text for part in cls._tokenize(node.application_context)
        ):
            return True

        # Verificação de atribuição canônica
        if node.canonical_context:
            return cls._check_canonical_context_compatibility(node, text)
        return False

    @classmethod
    def _check_canonical_context_compatibility(
        cls,
        node: ConceptNode,
        text: str,
    ) -> bool:
        """Verifica se o contexto canônico do conceito é compatível com o texto atual."""
        lower_text = text.lower()
        for context_key, context_value in node.canonical_context.items():
            # Verifica se o contexto canônico contém indicações de incompatibilidade
            if "NÃO" in context_value.upper():
                # Extrai termos proibidos (após "NÃO")
                not_parts = context_value.upper().split("NÃO")[1:]
                for not_part in not_parts:
                    prohibited_terms = cls._extract_prohibited_terms(not_part.strip())
                    if any(term in lower_text for term in prohibited_terms):
                        # Incompatível - gera alerta para L7
                        cls._generate_canonical_alert(node, context_key, context_value, text)
                        return False
            # Verifica se o contexto canônico requer termos específicos
            elif ":" in context_value:
                required_terms = cls._extract_required_terms(context_value)
                if any(term in lower_text for term in required_terms):
                    return True
        return True  # Compatível por padrão se não há restrições específicas

    @classmethod
    def _extract_prohibited_terms(cls, not_part: str) -> List[str]:
        """Extrai termos proibidos de uma parte 'NÃO ...'."""
        # Remove pontuação e divide por vírgulas ou 'ou'
        terms = re.split(r'[,\s]+ou[\s]+|[,;]', not_part)
        return [term.strip().lower() for term in terms if term.strip()]

    @classmethod
    def _extract_required_terms(cls, context_value: str) -> List[str]:
        """Extrai termos requeridos do contexto canônico."""
        # Assume formato "Fonte: descrição, termos requeridos"
        parts = context_value.split(":")
        if len(parts) > 1:
            description = parts[1].strip()
            terms = re.findall(r"[a-záàãâéêíóôõúüçA-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]+", description)
            return [term.lower() for term in terms if len(term) > 3]
        return []

    @classmethod
    def _generate_canonical_alert(
        cls,
        node: ConceptNode,
        context_key: str,
        context_value: str,
        text: str,
    ) -> None:
        """Gera um alerta de incompatibilidade canônica para ser passado ao L7."""
        # Armazena o alerta em uma variável global ou estrutura compartilhada
        # Por simplicidade, vamos usar um dicionário global para alertas
        if not hasattr(cls, '_canonical_alerts'):
            cls._canonical_alerts = []
        alert = {
            'concept': node.term,
            'canonical_context': f"{context_key}: {context_value}",
            'incompatible_usage': text[:100] + "..." if len(text) > 100 else text,
            'alert_type': 'canonical_incompatibility'
        }
        cls._canonical_alerts.append(alert)

    @classmethod
    def get_canonical_alerts(cls) -> List[Dict]:
        """Retorna e limpa os alertas canônicos gerados."""
        if not hasattr(cls, '_canonical_alerts'):
            cls._canonical_alerts = []
        alerts = cls._canonical_alerts[:]
        cls._canonical_alerts.clear()
        return alerts

    @classmethod
    def _extract_source_keywords(cls, source: str) -> List[str]:
        return [
            token for token in re.findall(r"[a-záàãâéêíóôõúüçA-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]+", source.lower())
            if len(token) > 3
        ]

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        return [token for token in re.findall(r"[a-záàãâéêíóôõúüçA-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]+", text.lower()) if len(token) > 3]

    @classmethod
    def _infer_relation(
        cls,
        node: ConceptNode,
        text: str,
        concepts: List[ConceptNode],
    ) -> str:
        domain_keywords = cls.CONTEXTUAL_KEYWORDS.get(node.domain, [])
        for keyword in domain_keywords:
            if keyword in text:
                return cls._build_context_sentence(node, keyword)

        related = cls._related_concepts(node, concepts, text)
        if related:
            return cls._build_related_context(node, related)

        return ""

    @classmethod
    def _related_concepts(
        cls,
        node: ConceptNode,
        concepts: List[ConceptNode],
        text: str,
    ) -> List[str]:
        related = []
        for other in concepts:
            if other.term == node.term:
                continue
            if other.term.lower() in text:
                related.append(other.term)
        return related

    @classmethod
    def _build_context_sentence(cls, node: ConceptNode, keyword: str) -> str:
        return (
            f"No contexto da pergunta, '{node.term}' é aplicado como um conceito de {node.domain}"
            f" relacionado a '{keyword}', indicando como o prompt utiliza seu significado prático."
        )

    @classmethod
    def _build_related_context(cls, node: ConceptNode, related: List[str]) -> str:
        related_terms = ", ".join(related[:3])
        return (
            f"Neste caso, '{node.term}' aparece em conjunto com {related_terms},"
            f" o que sugere seu papel prático na análise do prompt."
        )

    @classmethod
    def _default_context(cls, node: ConceptNode) -> str:
        return (
            f"No contexto atual, '{node.term}' representa {node.definition.lower()}"
            f" e serve como um conceito relevante para o problema expresso no prompt."
        )

    @classmethod
    def summarize_application_context(cls, concepts: List[ConceptNode]) -> str:
        parts = [node.application_context for node in concepts if node.application_context]
        return " ".join(parts)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split()).strip()
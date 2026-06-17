"""
RAG HÍBRIDO COM CONTEXT INJECTION
===================================
Camada de Retrieval-Augmented Generation (RAG) que trabalha de forma conjunta
com as camadas L1 e L2, usando um protocolo híbrido de:
  1. Context Injection (stuffing direto) — injeta contexto pré-selecionado
  2. Retrieval Seletivo por Domínios — busca documentos relevantes dinamicamente

A solução é HIBRIDA: injeção direta + retrieval seletivo baseado em domínios.
Integração com KB especializado (knowledge_base.py) e ChromaDB.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import re
from enum import Enum

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from knowledge_base import get_domain_knowledge_base, load_kb_from_file, merge_kb
except ImportError:
    get_domain_knowledge_base = None
    load_kb_from_file = None
    merge_kb = None


# ─────────────────────────────────────────────────────────────────────────────
# Enums e Estruturas de Dados
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalStrategy(Enum):
    """Estratégia de retrieval seletivo."""
    DIRECT_INJECTION = "direct_injection"          # Apenas contexto injetado
    SEMANTIC_RETRIEVAL = "semantic_retrieval"      # Busca semântica em ChromaDB
    HYBRID = "hybrid"                               # Injeção + Retrieval seletivo
    DOMAIN_AWARE = "domain_aware"                   # Retrieval baseado em domínio


@dataclass
class DomainContext:
    """Contexto especializado de um domínio."""
    domain_name: str
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    kb_path: str = ""                               # Caminho para KB do domínio
    chroma_collection: str = ""                     # Nome da coleção no ChromaDB
    system_prompt: str = ""                         # System prompt especializado
    injection_weight: float = 0.8                   # Peso da injeção direta [0,1]
    retrieval_weight: float = 0.2                   # Peso do retrieval [0,1]
    max_injected_docs: int = 3                      # Máx de docs injetados
    max_retrieved_docs: int = 5                     # Máx de docs recuperados


@dataclass
class RetrievedDocument:
    """Um documento recuperado do knowledge base."""
    content: str
    source: str = ""
    domain: str = ""
    relevance_score: float = 1.0
    is_injected: bool = False                       # Se vem de injeção direta
    metadata: Dict[str, Any] = field(default_factory=dict)

    def truncate(self, max_length: int = 500) -> str:
        """Trunca o conteúdo para não poluir o contexto."""
        if len(self.content) > max_length:
            return self.content[:max_length].rstrip() + "..."
        return self.content


@dataclass
class RAGContext:
    """Contexto híbrido compilado para injeção no prompt."""
    query: str
    domain: str = "geral"
    retrieved_documents: List[RetrievedDocument] = field(default_factory=list)
    injected_knowledge: Dict[str, float] = field(default_factory=dict)
    compiled_context: str = ""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    confidence_score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Sistema de Domínios Pré-configurados
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DOMAINS: Dict[str, DomainContext] = {
    "filosofia": DomainContext(
        domain_name="filosofia",
        description="Filosofia, epistemologia, lógica clássica",
        keywords=["conhecimento", "verdade", "ser", "essência", "substância", "silogismo"],
        kb_path="data/kb_filosofia.json",
        chroma_collection="filosofia_corpus",
        system_prompt="""Você é um especialista rigoroso em filosofia com acesso a uma base de conhecimento 
especializada em epistemologia, lógica e metafísica. Responda sempre usando o contexto fornecido quando 
relevante. Seja preciso, cite fontes filosóficas e mantenha o rigor conceitual.""",
        injection_weight=0.8,
        retrieval_weight=0.2,
    ),
    "lógica": DomainContext(
        domain_name="lógica",
        description="Lógica formal, lógica paraconsistente, teoria de modelos",
        keywords=["proposição", "predicado", "quantificador", "inferência", "validade", "contradição"],
        kb_path="data/kb_logica.json",
        chroma_collection="logica_corpus",
        system_prompt="""Você é um especialista em lógica formal e paraconsistência. Responda sempre 
usando o contexto fornecido quando relevante. Mantenha a precisão técnica, use notação apropriada e 
cite definições formais quando necessário.""",
        injection_weight=0.75,
        retrieval_weight=0.25,
    ),
    "epistemologia": DomainContext(
        domain_name="epistemologia",
        description="Epistemologia, teoria do conhecimento, justificação epistêmica",
        keywords=["justificação", "crença", "conhecimento", "evidência", "confiabilismo"],
        kb_path="data/kb_epistemologia.json",
        chroma_collection="epistemologia_corpus",
        system_prompt="""Você é um especialista rigoroso em epistemologia com acesso a uma base de 
conhecimento especializada. Responda sempre usando o contexto fornecido quando relevante. Cite teorias 
epistemológicas estabelecidas e seja preciso na caracterização de conceitos.""",
        injection_weight=0.8,
        retrieval_weight=0.2,
    ),
    "geral": DomainContext(
        domain_name="geral",
        description="Conhecimento geral e enciclopédico",
        keywords=[],
        kb_path="data/kb.json",
        chroma_collection="general_corpus",
        system_prompt="""Você é um especialista rigoroso com acesso a uma base de conhecimento especializada.
Responda sempre usando o contexto fornecido quando relevante. Seja preciso e cite fontes quando possível.""",
        injection_weight=0.7,
        retrieval_weight=0.3,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Motor RAG Híbrido com Context Injection
# ─────────────────────────────────────────────────────────────────────────────

class HybridRAGContextInjectionEngine:
    """
    Motor principal de RAG híbrido que combina:
    - Context Injection (injeção direta de KB/documentos pré-selecionados)
    - Semantic Retrieval (busca em ChromaDB por similaridade)
    - Domain-Aware Selection (seleção baseada em domínio)
    
    A estratégia HYBRID usa injeção como contexto de base + retrieval seletivo.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chroma_path: str = "chromadb",
        verbose: bool = True,
    ):
        self.config = config or {}
        self.embedding_model = embedding_model
        self.chroma_path = Path(chroma_path)
        self.verbose = verbose
        self.domains = dict(DEFAULT_DOMAINS)
        self.chroma_stores: Dict[str, Any] = {}  # Cache de lojas ChromaDB
        self._initialize_chroma()

    def _initialize_chroma(self) -> None:
        """Inicializa conexões com ChromaDB para cada domínio."""
        if not HAS_CHROMA:
            if self.verbose:
                print("[RAG] ChromaDB não disponível, usando apenas injeção direta.")
            return

        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            for domain_name in self.domains:
                chroma_dir = self.chroma_path / domain_name
                if chroma_dir.exists() and chroma_dir.is_dir():
                    try:
                        store = Chroma(
                            persist_directory=str(chroma_dir),
                            embedding_function=embeddings,
                            collection_name=self.domains[domain_name].chroma_collection,
                        )
                        self.chroma_stores[domain_name] = store
                        if self.verbose:
                            print(f"[RAG] ChromaDB carregado para domínio '{domain_name}'")
                    except Exception as e:
                        if self.verbose:
                            print(f"[RAG] Erro ao carregar ChromaDB para '{domain_name}': {e}")
        except Exception as e:
            if self.verbose:
                print(f"[RAG] Erro ao inicializar ChromaDB: {e}")

    def register_domain(self, domain: DomainContext) -> None:
        """Registra um novo domínio."""
        self.domains[domain.domain_name] = domain

    def detect_domain(self, query: str, concepts: Optional[List[str]] = None) -> Tuple[str, float]:
        """
        Detecta qual domínio é mais relevante para a query usando keywords matching.
        Retorna (domain_name, confidence_score).
        """
        query_lower = query.lower()
        scores = {}

        for domain_name, domain_ctx in self.domains.items():
            score = 0.0
            if domain_ctx.keywords:
                for kw in domain_ctx.keywords:
                    if kw.lower() in query_lower:
                        score += 1.0
            if concepts:
                for concept in concepts:
                    if concept.lower() in query_lower:
                        score += 0.5

            scores[domain_name] = score

        # Normaliza scores
        max_score = max(scores.values()) if scores else 0.0
        if max_score > 0:
            best_domain = max(scores, key=scores.get)
            confidence = scores[best_domain] / (max_score + 1)
        else:
            best_domain = "geral"
            confidence = 0.1

        return best_domain, confidence

    def get_injected_knowledge(
        self,
        domain: str,
        query: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Recupera conhecimento para injeção direta do KB do domínio.
        Usa get_domain_knowledge_base se disponível.
        """
        if not get_domain_knowledge_base:
            return {}

        try:
            kb = get_domain_knowledge_base(
                domain=domain,
                config=self.config,
                query_for_rag=query,
            )
            return kb
        except Exception as e:
            if self.verbose:
                print(f"[RAG] Erro ao recuperar KB do domínio '{domain}': {e}")
            return {}

    def retrieve_documents(
        self,
        query: str,
        domain: str = "geral",
        k: int = 5,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
    ) -> List[RetrievedDocument]:
        """
        Recupera documentos relevantes usando a estratégia especificada.
        
        Strategies:
        - DIRECT_INJECTION: Sem retrieval, apenas contexto injetado
        - SEMANTIC_RETRIEVAL: Apenas busca em ChromaDB
        - HYBRID: Injeção + Retrieval seletivo
        - DOMAIN_AWARE: Retrieval específico do domínio
        """
        results: List[RetrievedDocument] = []

        if strategy == RetrievalStrategy.DIRECT_INJECTION:
            # Apenas contexto injetado, sem retrieval dinâmico
            return results

        domain_ctx = self.domains.get(domain, self.domains["geral"])
        max_injected = domain_ctx.max_injected_docs
        max_retrieved = domain_ctx.max_retrieved_docs

        # ─────────────────────────────────────────────────────────────────────
        # Estratégia HYBRID: Injeção + Retrieval seletivo
        # ─────────────────────────────────────────────────────────────────────
        if strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.DOMAIN_AWARE):
            # Etapa 1: Contexto injetado (KB direto)
            injected_kb = self.get_injected_knowledge(domain, query)
            if injected_kb:
                # Seleciona top-k termos por relevância
                sorted_terms = sorted(injected_kb.items(), key=lambda x: x[1], reverse=True)
                for i, (term, score) in enumerate(sorted_terms[:max_injected]):
                    results.append(
                        RetrievedDocument(
                            content=f"Termo: {term}",
                            source=f"KB-{domain}",
                            domain=domain,
                            relevance_score=float(score),
                            is_injected=True,
                            metadata={"type": "kb_term", "weight": score},
                        )
                    )

        # Etapa 2: Retrieval semântico (ChromaDB)
        if strategy in (RetrievalStrategy.SEMANTIC_RETRIEVAL, RetrievalStrategy.HYBRID):
            if domain in self.chroma_stores:
                try:
                    chroma = self.chroma_stores[domain]
                    docs = chroma.similarity_search(query, k=max_retrieved)
                    for doc in docs:
                        # Extrai score se disponível
                        score = getattr(doc, "metadata", {}).get("score", 0.8)
                        results.append(
                            RetrievedDocument(
                                content=doc.page_content if hasattr(doc, "page_content") else str(doc),
                                source=f"ChromaDB-{domain}",
                                domain=domain,
                                relevance_score=float(score),
                                is_injected=False,
                                metadata=getattr(doc, "metadata", {}),
                            )
                        )
                except Exception as e:
                    if self.verbose:
                        print(f"[RAG] Erro ao recuperar de ChromaDB-{domain}: {e}")

        # Ordena por relevância
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:k]

    def compile_context(
        self,
        query: str,
        retrieved_docs: List[RetrievedDocument],
        injected_kb: Optional[Dict[str, float]] = None,
        domain: str = "geral",
        include_system_prompt: bool = True,
    ) -> RAGContext:
        """
        Compila o contexto final para injeção no prompt.
        Combina documentos recuperados, KB injetado e system prompt.
        """
        domain_ctx = self.domains.get(domain, self.domains["geral"])
        lines = []

        # ─────────────────────────────────────────────────────────────────────
        # Parte 1: System Prompt especializado
        # ─────────────────────────────────────────────────────────────────────
        if include_system_prompt and domain_ctx.system_prompt:
            lines.append("## Instruções do Sistema")
            lines.append(domain_ctx.system_prompt)
            lines.append("")

        # ─────────────────────────────────────────────────────────────────────
        # Parte 2: Documentos Injetados (Context Injection)
        # ─────────────────────────────────────────────────────────────────────
        injected_docs = [d for d in retrieved_docs if d.is_injected]
        if injected_docs:
            lines.append("## Contexto Base Injetado (Domínio)")
            for doc in injected_docs:
                lines.append(f"- **{doc.source}** [{doc.relevance_score:.2f}]: {doc.truncate()}")
            lines.append("")

        # ─────────────────────────────────────────────────────────────────────
        # Parte 3: Documentos Recuperados (Semantic Retrieval)
        # ─────────────────────────────────────────────────────────────────────
        retrieved_only = [d for d in retrieved_docs if not d.is_injected]
        if retrieved_only:
            lines.append("## Contexto Recuperado (ChromaDB)")
            for doc in retrieved_only:
                lines.append(f"- **{doc.source}**: {doc.truncate()}")
            lines.append("")

        # ─────────────────────────────────────────────────────────────────────
        # Parte 4: Knowledge Base Terms (se fornecido)
        # ─────────────────────────────────────────────────────────────────────
        if injected_kb:
            lines.append("## Termos-Chave do Knowledge Base")
            sorted_terms = sorted(injected_kb.items(), key=lambda x: x[1], reverse=True)[:10]
            for term, score in sorted_terms:
                lines.append(f"- {term}: {score:.2f}")
            lines.append("")

        # ─────────────────────────────────────────────────────────────────────
        # Parte 5: Instrução de Resposta
        # ─────────────────────────────────────────────────────────────────────
        lines.append("## Pergunta do Usuário")
        lines.append(f"{query}")
        lines.append("")
        lines.append("---")
        lines.append("Baseando-se no contexto injetado e recuperado acima, elabore uma resposta rigorosa.")
        lines.append("")

        compiled = "\n".join(lines)

        # Calcula confidence score
        conf = 0.0
        if injected_docs:
            conf += domain_ctx.injection_weight * (sum(d.relevance_score for d in injected_docs) / len(injected_docs))
        if retrieved_only:
            conf += domain_ctx.retrieval_weight * (sum(d.relevance_score for d in retrieved_only) / len(retrieved_only))
        conf = min(1.0, conf)

        return RAGContext(
            query=query,
            domain=domain,
            retrieved_documents=retrieved_docs,
            injected_knowledge=injected_kb or {},
            compiled_context=compiled,
            strategy=RetrievalStrategy.HYBRID,
            confidence_score=conf,
        )

    def process(
        self,
        query: str,
        concepts: Optional[List[str]] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        k: int = 8,
        auto_detect_domain: bool = True,
    ) -> RAGContext:
        """
        Pipeline completo de RAG híbrido.
        
        Etapas:
        1. Detecta domínio (se auto_detect_domain=True)
        2. Recupera documentos (injeção + retrieval)
        3. Compila contexto final
        4. Retorna RAGContext pronto para injeção
        """
        # Detecta domínio
        if auto_detect_domain:
            domain, conf = self.detect_domain(query, concepts)
            if self.verbose:
                print(f"[RAG] Domínio detectado: {domain} (confiança: {conf:.2f})")
        else:
            domain = "geral"

        # Recupera conhecimento injetado
        injected_kb = self.get_injected_knowledge(domain, query)

        # Recupera documentos
        retrieved_docs = self.retrieve_documents(
            query=query,
            domain=domain,
            k=k,
            strategy=strategy,
        )

        # Compila contexto
        rag_context = self.compile_context(
            query=query,
            retrieved_docs=retrieved_docs,
            injected_kb=injected_kb,
            domain=domain,
            include_system_prompt=True,
        )

        return rag_context

    def format_for_l1_l2(self, rag_context: RAGContext) -> Dict[str, Any]:
        """
        Formata o contexto RAG para consumo pelas camadas L1 (Conceitos) e L2 (Juízos).
        Retorna um dicionário com:
        - domain: domínio detectado
        - injected_context: string do contexto injetado
        - kb_terms: dicionário termo -> score
        - system_prompt: system prompt especializado
        - documents: lista de documentos
        """
        domain_ctx = self.domains.get(rag_context.domain, self.domains["geral"])

        return {
            "domain": rag_context.domain,
            "injected_context": rag_context.compiled_context,
            "kb_terms": rag_context.injected_knowledge,
            "system_prompt": domain_ctx.system_prompt,
            "documents": [
                {
                    "content": doc.truncate(1000),
                    "source": doc.source,
                    "relevance": doc.relevance_score,
                    "is_injected": doc.is_injected,
                }
                for doc in rag_context.retrieved_documents
            ],
            "confidence": rag_context.confidence_score,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Funções Auxiliares de Alto Nível
# ─────────────────────────────────────────────────────────────────────────────

def create_hybrid_rag_engine(
    config: Optional[Dict[str, Any]] = None,
    chroma_path: str = "chromadb",
) -> HybridRAGContextInjectionEngine:
    """Factory para criar uma instância do motor RAG."""
    return HybridRAGContextInjectionEngine(config=config, chroma_path=chroma_path)


def process_query_with_rag(
    query: str,
    concepts: Optional[List[str]] = None,
    domain: Optional[str] = None,
    auto_detect: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> RAGContext:
    """
    Função de conveniência para processar uma query com RAG híbrido.
    
    Exemplo:
        rag_ctx = process_query_with_rag("O que é conhecimento?", domain="epistemologia")
        print(rag_ctx.compiled_context)
    """
    engine = create_hybrid_rag_engine(config=config)
    return engine.process(
        query=query,
        concepts=concepts,
        auto_detect_domain=auto_detect,
        strategy=RetrievalStrategy.HYBRID,
    )

"""
EXEMPLO DE USO: RAG Híbrido com L1-L2
======================================
Demonstra como usar o sistema completo:
- RAG Híbrido (Context Injection + Retrieval Seletivo)
- Integração com L1 (Conceitos) e L2 (Juízos Kantianos)
- Domain-aware Knowledge Base
- Context Injection com system_prompt especializado
"""

from pathlib import Path
import json
from typing import Optional, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# 1. EXEMPLO BÁSICO: Apenas RAG Híbrido
# ─────────────────────────────────────────────────────────────────────────────

def example_1_basic_rag():
    """
    Exemplo 1: Usa RAG híbrido para processar uma query.
    Demonstra: Context Injection + Retrieval Seletivo.
    """
    print("\n" + "="*70)
    print("EXEMPLO 1: RAG Híbrido Básico")
    print("="*70)

    from rag_hybrid_context_injection import (
        HybridRAGContextInjectionEngine,
        RetrievalStrategy,
    )

    # Cria motor RAG
    rag_engine = HybridRAGContextInjectionEngine(verbose=True)

    # Query de teste
    query = "O que é verdade em lógica paraconsistente?"

    # Processa com RAG híbrido
    rag_context = rag_engine.process(
        query=query,
        strategy=RetrievalStrategy.HYBRID,
        auto_detect_domain=True,
    )

    print(f"\n✓ Domínio detectado: {rag_context.domain}")
    print(f"✓ Confiança: {rag_context.confidence_score:.2%}")
    print(f"✓ Documentos recuperados: {len(rag_context.retrieved_documents)}")
    print(f"\n--- Contexto Compilado (para injeção no LLM) ---")
    print(rag_context.compiled_context[:500] + "...")

    return rag_context


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXEMPLO: RAG + L1-L2 Pipeline Completo
# ─────────────────────────────────────────────────────────────────────────────

def example_2_l1_l2_rag_pipeline():
    """
    Exemplo 2: Pipeline completo L1-L2-RAG.
    Demonstra: Conceitos + Juízos + RAG Híbrido integrados.
    """
    print("\n" + "="*70)
    print("EXEMPLO 2: Pipeline Completo L1-L2-RAG")
    print("="*70)

    from l1_l2_rag_integration import create_l1_l2_rag_pipeline

    # Cria pipeline
    pipeline = create_l1_l2_rag_pipeline()

    # Query de teste
    query = "Qual é a definição epistemológica de conhecimento justificado?"

    # Processa
    result = pipeline.process(query)

    print(f"\n✓ Domínio: {result['domain']}")
    print(f"✓ Confiança: {result['confidence']:.2%}")
    print(f"✓ Conceitos (L1): {len(result['l1_output'].concepts)}")
    print(f"✓ Juízos (L2): {len(result['l2_output'].judgments)}")

    print(f"\n--- L1 (Conceitos) ---")
    for concept in result['l1_output'].concepts[:3]:
        term = concept.term if hasattr(concept, "term") else str(concept)
        print(f"  • {term}")

    print(f"\n--- L2 (Top Judgment) ---")
    if result['l2_output'].top_judgment:
        print(f"  {str(result['l2_output'].top_judgment)[:200]}...")

    print(f"\n--- Contexto Compilado (para injeção) ---")
    print(result['compiled_context'][:600] + "...")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXEMPLO: Domain Detection e Context Injection Seletiva
# ─────────────────────────────────────────────────────────────────────────────

def example_3_domain_specific_injection():
    """
    Exemplo 3: Detecta domínio automaticamente e usa system_prompt especializado.
    Demonstra: Domain-aware context injection.
    """
    print("\n" + "="*70)
    print("EXEMPLO 3: Domain-Specific Context Injection")
    print("="*70)

    from rag_hybrid_context_injection import HybridRAGContextInjectionEngine

    rag_engine = HybridRAGContextInjectionEngine(verbose=True)

    # Queries de diferentes domínios
    queries = [
        ("Aristóteles define a substância como categoria fundamental", "filosofia"),
        ("Na lógica paraconsistente, é possível ter P e ¬P simultaneamente?", "lógica"),
        ("Como a justificação interna diferencia-se da justificação externa?", "epistemologia"),
    ]

    for query, expected_domain in queries:
        print(f"\n--- Query: {query[:60]}... ---")

        rag_context = rag_engine.process(
            query=query,
            auto_detect_domain=True,
        )

        detected = rag_context.domain
        match = "✓" if detected == expected_domain else "✗"
        print(f"{match} Domínio detectado: {detected} (esperado: {expected_domain})")

        # Mostra system_prompt especializado
        domain_ctx = rag_engine.domains[detected]
        print(f"\nSystem Prompt (domínio {detected}):")
        print(f"  {domain_ctx.system_prompt[:150]}...")


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXEMPLO: Retrieval Strategy Comparativo
# ─────────────────────────────────────────────────────────────────────────────

def example_4_strategy_comparison():
    """
    Exemplo 4: Compara diferentes estratégias de retrieval.
    Demonstra: Injeção direta vs Semantic Retrieval vs Hybrid.
    """
    print("\n" + "="*70)
    print("EXEMPLO 4: Estratégias de Retrieval Comparativas")
    print("="*70)

    from rag_hybrid_context_injection import (
        HybridRAGContextInjectionEngine,
        RetrievalStrategy,
    )

    rag_engine = HybridRAGContextInjectionEngine(verbose=False)
    query = "O que é uma proposição numa lógica não-clássica?"

    strategies = [
        RetrievalStrategy.DIRECT_INJECTION,
        RetrievalStrategy.SEMANTIC_RETRIEVAL,
        RetrievalStrategy.HYBRID,
    ]

    for strategy in strategies:
        rag_context = rag_engine.process(
            query=query,
            strategy=strategy,
            auto_detect_domain=True,
        )

        injected = sum(1 for d in rag_context.retrieved_documents if d.is_injected)
        retrieved = len(rag_context.retrieved_documents) - injected

        print(f"\n--- Estratégia: {strategy.value} ---")
        print(f"  Injetados: {injected}")
        print(f"  Recuperados: {retrieved}")
        print(f"  Confiança: {rag_context.confidence_score:.2%}")
        print(f"  Contexto (chars): {len(rag_context.compiled_context)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXEMPLO AVANÇADO: Formatação para LLM com System Prompt Customizado
# ─────────────────────────────────────────────────────────────────────────────

def example_5_llm_formatted_output():
    """
    Exemplo 5: Formata saída para injeção direta em LLM local.
    Demonstra: Context Injection com system_prompt customizado.
    """
    print("\n" + "="*70)
    print("EXEMPLO 5: Formatação para Injeção em LLM")
    print("="*70)

    from l1_l2_rag_integration import create_l1_l2_rag_pipeline

    pipeline = create_l1_l2_rag_pipeline()

    query = "Explique a diferença entre conhecimento e opinião justificada."
    result = pipeline.process(query)

    # System prompt customizado (fornecido pelo usuário)
    system_prompt_custom = """Você é um especialista rigoroso com acesso a uma base de conhecimento especializada.
Responda sempre usando o contexto fornecido quando relevante. Seja preciso e cite fontes quando possível."""

    # Monta mensagens para LLM
    messages = [
        {
            "role": "system",
            "content": system_prompt_custom,
        },
        {
            "role": "user",
            "content": result["compiled_context"],
        },
    ]

    print("\n--- Mensagens formatadas para LLM (JSON) ---")
    print(json.dumps(messages, indent=2, ensure_ascii=False)[:800] + "...")

    print("\n--- Como usar com Ollama local ---")
    print("""
    import ollama

    response = ollama.chat(
        model="doninha8:latest",
        messages=[{"role": "user", "content": result["compiled_context"]}],
        stream=False,
        options={
            "temperature": 0.3,
            "num_ctx": 8192,
        },
    )
    if isinstance(response, dict):
        print(response.get("message", {}).get("content", ""))
    else:
        print(response)
    """)

    return messages


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXEMPLO: Criando domínios customizados
# ─────────────────────────────────────────────────────────────────────────────

def example_6_custom_domains():
    """
    Exemplo 6: Cria e registra domínios customizados.
    Demonstra: Extensibilidade do sistema.
    """
    print("\n" + "="*70)
    print("EXEMPLO 6: Domínios Customizados")
    print("="*70)

    from rag_hybrid_context_injection import (
        HybridRAGContextInjectionEngine,
        DomainContext,
    )

    rag_engine = HybridRAGContextInjectionEngine(verbose=True)

    # Define novo domínio customizado
    domain_direito = DomainContext(
        domain_name="direito",
        description="Direito civil, constitucional e penal",
        keywords=["lei", "código", "artigo", "direito", "obrigação", "contrato"],
        kb_path="data/kb_direito.json",
        chroma_collection="direito_corpus",
        system_prompt="""Você é um especialista em direito com rigorosa base legal.
Cite sempre artigos, precedentes e legislação pertinente. Mantenha precisão técnica e referencias às leis.""",
        injection_weight=0.85,
        retrieval_weight=0.15,
    )

    # Registra domínio
    rag_engine.register_domain(domain_direito)

    print(f"\n✓ Domínio 'direito' registrado")
    print(f"  Keywords: {', '.join(domain_direito.keywords[:3])}...")
    print(f"  System Prompt: {domain_direito.system_prompt[:100]}...")

    # Testa com query de direito
    query = "Qual é o prazo para prescrição de débitos fiscais?"
    detected_domain, conf = rag_engine.detect_domain(query)
    print(f"\n✓ Query sobre direito detectado como: {detected_domain}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PIPELINE COMPLETO DE PONTA A PONTA
# ─────────────────────────────────────────────────────────────────────────────

def example_7_end_to_end_pipeline():
    """
    Exemplo 7: Pipeline completo de ponta a ponta.
    Demonstra: Fluxo completo desde query até resposta estruturada.
    """
    print("\n" + "="*70)
    print("EXEMPLO 7: Pipeline de Ponta a Ponta")
    print("="*70)

    from l1_l2_rag_integration import create_l1_l2_rag_pipeline

    # Cria pipeline
    pipeline = create_l1_l2_rag_pipeline()

    # Query
    query = "Como Kant define o juízo analítico?"

    print(f"\n[1] Input: {query}")

    # Etapa 1: Processamento completo
    result = pipeline.process(query)

    print(f"\n[2] Domain Detection")
    print(f"    ✓ Domain: {result['domain']}")
    print(f"    ✓ Confidence: {result['confidence']:.2%}")

    print(f"\n[3] L1 (Conceitos) - Extração e Enriquecimento")
    print(f"    ✓ Conceitos extraídos: {len(result['l1_output'].concepts)}")
    for i, c in enumerate(result['l1_output'].concepts[:3], 1):
        print(f"      {i}. {c.term if hasattr(c, 'term') else str(c)}")

    print(f"\n[4] L2 (Juízos Kantianos) - Análise e Enriquecimento")
    print(f"    ✓ Juízos gerados: {len(result['l2_output'].judgments)}")
    if result['l2_output'].top_judgment:
        judgment_str = str(result['l2_output'].top_judgment)
        print(f"    ✓ Top Judgment: {judgment_str[:120]}...")

    print(f"\n[5] Context Injection")
    print(f"    ✓ System Prompt: {result['system_prompt'][:100]}...")
    print(f"    ✓ Contexto compilado: {len(result['compiled_context'])} caracteres")

    print(f"\n[6] Saída Final (pronta para LLM)")
    print(f"    ✓ RAG Context Summary: {result['rag_context_summary']}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: Executa todos os exemplos
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Executa todos os exemplos."""
    print("\n" + "="*70)
    print("DEMONSTRAÇÃO: RAG Híbrido com Context Injection (L1-L2)")
    print("="*70)

    try:
        # Exemplo 1: RAG básico
        example_1_basic_rag()
    except Exception as e:
        print(f"\n❌ Exemplo 1 falhou: {e}")

    try:
        # Exemplo 2: L1-L2-RAG pipeline
        example_2_l1_l2_rag_pipeline()
    except Exception as e:
        print(f"\n❌ Exemplo 2 falhou: {e}")

    try:
        # Exemplo 3: Domain-specific injection
        example_3_domain_specific_injection()
    except Exception as e:
        print(f"\n❌ Exemplo 3 falhou: {e}")

    try:
        # Exemplo 4: Strategy comparison
        example_4_strategy_comparison()
    except Exception as e:
        print(f"\n❌ Exemplo 4 falhou: {e}")

    try:
        # Exemplo 5: LLM formatted output
        example_5_llm_formatted_output()
    except Exception as e:
        print(f"\n❌ Exemplo 5 falhou: {e}")

    try:
        # Exemplo 6: Custom domains
        example_6_custom_domains()
    except Exception as e:
        print(f"\n❌ Exemplo 6 falhou: {e}")

    try:
        # Exemplo 7: End-to-end
        example_7_end_to_end_pipeline()
    except Exception as e:
        print(f"\n❌ Exemplo 7 falhou: {e}")

    print("\n" + "="*70)
    print("✓ Demonstração concluída!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

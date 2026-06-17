#!/usr/bin/env python3
"""
Teste da classificação epistemológica BERT em L2 com (T, I, F).

Demonstra como o juízo assertórico é classificado segundo:
  T + F > 1 → paraconsistência
  T + I + F < 1 → incompletude
  I high → vagueza
  T high, I low, F low → assertiva confiante
"""

from l1_concept_table import ConceptTable
from l2_kantian_judgments import KantianJudgmentEngine, BERTAssertionClassifier


def main():
    print("=" * 70)
    print("TESTE: Classificação Epistemológica em L2 (T, I, F)")
    print("=" * 70)

    # Inicializa as camadas L1 e L2
    concept_table = ConceptTable()
    kant_engine = KantianJudgmentEngine(concept_table)

    # Prompts de teste
    test_prompts = [
        "água quente verdadeira",
        "pode ser falso e verdadeiro ao mesmo tempo",
        "indeterminado e indefinido",
        "sempre verdadeiro",
        "contraditório e incompleteto",
    ]

    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        print("-" * 70)

        # L1: Extração de conceitos
        concepts = concept_table.extract_concepts(prompt, llm_context=prompt)
        print(f"  L1 Conceitos extraídos: {len(concepts)}")
        for concept in concepts[:3]:
            print(f"    • {concept.term} [{concept.domain}]")
            if concept.application_context:
                print(f"      Contexto: {concept.application_context[:60]}...")

        # L2: Juízos kantianos com classificação epistemológica
        judgments = kant_engine.refine(prompt, concepts)
        print(f"\n  L2 Juízos assertóricos com (T, I, F):")

        # Filtra apenas juízos assertóricos para mostrar classificação
        assertoric_judgments = [j for j in judgments if j.modalidade == "Assertórico"]
        for i, judgment in enumerate(assertoric_judgments[:5], 1):
            ec = judgment.epistemic_classification
            print(f"\n    {i}. [{judgment.quantidade}/{judgment.qualidade}]")
            print(f"       Proposição: {judgment.proposicao[:55]}...")
            print(f"       {ec}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

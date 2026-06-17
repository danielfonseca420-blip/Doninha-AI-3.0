"""
PIPELINE PRINCIPAL — Modelo Híbrido de LLM
===========================================
Orquestra as 10 etapas do fluxo completo:

  1. Recepção do prompt
  2. Extração de conceitos [L1]
  3. Refinamento por Juízos Kantianos [L2]
  4. Silogismo Científico + Hempel
  5. Falseabilidade de Popper
  6. Avaliação Paraconsistente [L3]
  7. Síntese por Equivalência [L4]
  8. Geração da Resposta [L5 — opcional]
  9. Resposta Final em Texto Fluida [L6]
 10. Texto Final Definitivo [L7]

Usa config_loader, knowledge_base (KB escalável + RAG opcional), l5_generation
e opcionalmente o agente de pesquisa para enriquecer contexto.
"""

from __future__ import annotations
import sys
import re
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

from neural_truth_model import TruthScoringModel, load_tokenizer
from l1_concept_table import ConceptTable, ConceptNode, LogicLMSymbolicSolver
from l2_kantian_judgments import KantianJudgmentEngine, KantianJudgment
from syllogism_module import ScientificSyllogismPipeline
from l3_paraconsistent import ParaconsistentEngine, ParaconsistentValue
from l4_synthesis import RussellianSynthesisEngine, SynthesisResult
from l6_final_response import EpistemicContext, FinalResponseEngine
from l7_final_text import FinalTextEngine

try:
    from l4_russell_equivalence import load_concept_base
except Exception:
    load_concept_base = None  # type: ignore

try:
    from config_loader import load_config, PROJECT_ROOT
except Exception:
    load_config = None  # type: ignore
    PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from knowledge_base import get_knowledge_base, SEED_KNOWLEDGE_BASE
except Exception:
    get_knowledge_base = None  # type: ignore
    SEED_KNOWLEDGE_BASE = {}

try:
    from l5_generation import generate_response as l5_generate
except Exception:
    l5_generate = None  # type: ignore

try:
    from agente_busca_web import run_search_for_context
except Exception:
    run_search_for_context = None  # type: ignore

try:
    from agente_sintese_final import synthesize_final_text as synthesize_final_agent
except Exception:
    synthesize_final_agent = None  # type: ignore


def _get_kb(config: Optional[Dict[str, Any]], prompt: str, use_agent: bool) -> Dict[str, float]:
    if get_knowledge_base is None:
        return dict(SEED_KNOWLEDGE_BASE) if SEED_KNOWLEDGE_BASE else {}
    return get_knowledge_base(
        config=config,
        query_for_rag=prompt if use_agent else None,
    )


class HybridLLMPipeline:
    """
    Pipeline completo do Modelo Híbrido de LLM.
    Suporta config, KB escalável, L5 (geração), agente opcional e chat.
    """

    def __init__(
        self,
        knowledge_base: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> None:
        self._config = config or (load_config() if load_config else {})
        self.kb = knowledge_base or _get_kb(self._config, "", False)
        if not self.kb:
            self.kb = dict(SEED_KNOWLEDGE_BASE) if SEED_KNOWLEDGE_BASE else {}
        self.verbose = verbose

        self.L1 = ConceptTable()
        self.L2 = KantianJudgmentEngine(self.L1)
        self.SYL = ScientificSyllogismPipeline()

        # L3
        l3_cfg = self._config.get("l3", {})
        model_path = l3_cfg.get("model_path", "truth_scoring_model.pt")
        backbone_name = l3_cfg.get("backbone", "bert-base-multilingual-cased")
        if not Path(model_path).is_absolute():
            model_path = str(PROJECT_ROOT / model_path)
        neural_model = None
        neural_tokenizer = None
        if os.path.exists(model_path):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                neural_tokenizer = load_tokenizer(backbone_name)
                neural_model = TruthScoringModel(backbone_name=backbone_name)
                state = torch.load(model_path, map_location=device)
                neural_model.load_state_dict(state)
                neural_model.to(device)
                if self.verbose:
                    print(f"[L3] Modelo neural carregado de '{model_path}'")
                self.L3 = ParaconsistentEngine(neural_model=neural_model, neural_tokenizer=neural_tokenizer, device=device)
            except Exception as exc:
                if self.verbose:
                    print(f"[L3] Falha ao carregar modelo neural: {exc}")
                self.L3 = ParaconsistentEngine()
        else:
            self.L3 = ParaconsistentEngine()

        # L4
        russell_base = None
        rpath = self._config.get("l4", {}).get("russell_concepts_path", "l4_russell_concepts.json")
        if not Path(rpath).is_absolute():
            rpath = str(PROJECT_ROOT / rpath)
        if load_concept_base and os.path.exists(rpath):
            try:
                russell_base = load_concept_base(rpath)
                if self.verbose:
                    print("[L4] Base russelliana carregada.")
            except Exception:
                pass
        if russell_base is None and load_concept_base:
            try:
                from l4_russell_equivalence import build_russell_concept_base
                russell_base = build_russell_concept_base()
            except Exception:
                pass
        self.L4 = RussellianSynthesisEngine(
            self.kb,
            russell_concept_base=russell_base,
            use_concept_based_weights=(russell_base is not None),
            verification_config=self._config.get("l4_chain_verification", {}),
        )
        self.L6 = FinalResponseEngine()
        self.L7 = FinalTextEngine(config=self._config)  # Passa config para suportar múltiplos providers

    def _infer_domain(self, concepts: List[ConceptNode]) -> str:
        """Inferência simples de domínio majoritário a partir dos conceitos extraídos."""
        if not concepts:
            return "geral"
        domain_counts = {}
        for concept in concepts:
            domain = concept.domain.lower().strip() if concept.domain else "geral"
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return max(domain_counts, key=domain_counts.get)

    def _collect_canonical_sources(self, concepts: List[ConceptNode]) -> List[str]:
        sources = []
        seen = set()
        for concept in concepts:
            source = (concept.canonical_source or "").strip()
            if source and source not in seen:
                seen.add(source)
                sources.append(source)
        return sources

    def _summarize_judgments(self, judgments: List[KantianJudgment]) -> str:
        parts = []
        for idx, judgment in enumerate(judgments[:5], start=1):
            cls = getattr(judgment.epistemic_classification, "classification", "não_classificado")
            truth = getattr(judgment.epistemic_classification, "truth", 0.0)
            ind = getattr(judgment.epistemic_classification, "indeterminacy", 0.0)
            fals = getattr(judgment.epistemic_classification, "falsity", 0.0)
            parts.append(
                f"L2-{idx}: {judgment.proposicao[:120]} | pri={judgment.prioridade:.2f} | class={cls} | T/I/F={truth:.2f}/{ind:.2f}/{fals:.2f}"
            )
        return " ; ".join(parts) if parts else "nenhum juízo L2 disponível"

    def _summarize_paraconsistent(self, pv_list: List[ParaconsistentValue]) -> str:
        parts = []
        for idx, pv in enumerate(pv_list[:5], start=1):
            parts.append(
                f"L3-{idx}: μ={pv.mu:.3f} λ={pv.lam:.3f} state={pv.state} truth={pv.truth_value:.3f} certainty={pv.certainty:+.3f} contradiction={pv.contradiction:+.3f}"
            )
        return " ; ".join(parts) if parts else "nenhuma avaliação L3 disponível"

    def _build_citation_note(self, concepts: List[ConceptNode], agent_context: str) -> str:
        sources = self._collect_canonical_sources(concepts)
        if sources:
            return "Fontes bibliográficas/canonicais disponíveis para auditoria: " + "; ".join(sources[:6])
        if agent_context:
            return "Aviso de auditoria: a base local não expôs fontes bibliográficas verificáveis; a correspondência foi analisada com contexto RAG externo e sem citações canônicas confirmadas."
        return "Aviso de auditoria: nenhuma fonte bibliográfica verificável foi detectada na base local; a resposta foi produzida com base interna e deve ser tratada como não corroborada por referências externas."

    def _append_audit_block(self, text: str, label: str, details: str) -> str:
        block = f"\n\n[AUDIT {label}] {details}"
        return (text + block).strip()

    def process(
        self,
        prompt: str,
        chat_session: Optional[Any] = None,
        use_agent: Optional[bool] = None,
        skip_l5: bool = False,
        skip_l6: bool = False,
    ) -> SynthesisResult:
        """Executa o pipeline e retorna SynthesisResult (com response já gerada por L5 se ativo)."""
        t0 = time.perf_counter()
        use_agent = use_agent if use_agent is not None else self._config.get("agent", {}).get("use_agent", False)
        if chat_session and hasattr(chat_session, "get_context_for_prompt"):
            prompt_for_kb = chat_session.get_context_for_prompt(prompt, self._config.get("chat", {}).get("max_turns_in_context", 10))
        else:
            prompt_for_kb = prompt

        # KB pode ser enriquecido por RAG (Chroma) quando use_agent
        if use_agent and get_knowledge_base:
            self.kb = _get_kb(self._config, prompt_for_kb, True)
            if not self.kb:
                self.kb = dict(SEED_KNOWLEDGE_BASE) if SEED_KNOWLEDGE_BASE else {}

        self._log("\n" + "═" * 60)
        self._log(f"  PROMPT: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        self._log("═" * 60)

        limit = RussellianSynthesisEngine.check_fundamental_limits(prompt)
        if limit:
            self._log(f"\n{limit}")

        self._log("\n[ETAPA 2] L1 — Extração de Conceitos")
        concepts: List[ConceptNode] = self.L1.extract_concepts(prompt, llm_context=prompt_for_kb, domain="geral", config=self._config)
        domain = self._infer_domain(concepts)
        if domain != "geral":
            # Re-extrai com domínio específico para enriquecer com KB do domínio
            concepts = self.L1.extract_concepts(prompt, llm_context=prompt_for_kb, domain=domain, config=self._config)
        concepts_summary = ""
        if self.verbose and concepts:
            for c in concepts:
                syns = ", ".join(c.synonyms[:2]) or "—"
                self._log(f"  • {c.term:15s} | sinônimos: {syns}")
            concepts_summary = "; ".join(f"{c.term}({', '.join(c.synonyms[:2])})" for c in concepts[:8])

        self._log("\n[ETAPA 3] L2 — Juízos Kantianos")
        judgments: List[KantianJudgment] = self.L2.refine(prompt, concepts)
        top_judgments = ""
        if judgments:
            top_judgments = "\n".join(j.proposicao for j, _ in list(zip(judgments, [None] * 6))[:6])

        self._log("\n[ETAPAS 4+5] Silogismo + Hempel + Popper")
        prompt_terms = set(re.findall(r"[a-záàãâéêíóôõúüçA-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]+", prompt.lower()))
        kb_scores = {j.proposicao[:30]: self.kb.get(j.proposicao.split()[0], 0.3) for j in judgments}
        filtered = self.SYL.run(judgments, prompt_terms, kb_scores)
        self._log(f"  {len(judgments)} hipóteses → {len(filtered)} após filtros")

        self._log("\n[ETAPA 6] L3 — Lógica Paraconsistente + Classificação Epistemológica L2")
        props_with_priority = [(j.proposicao, score) for j, score in filtered]
        pv_list: List[ParaconsistentValue] = self.L3.evaluate(props_with_priority, self.kb)
        consistent = self.L3.check_global_consistency(pv_list)
        self._log(f"  Consistência global: {'✓' if consistent else '✗'}")

        epistemic_context = EpistemicContext(
            proposition_states=[
                {
                    "proposition": pv.proposition,
                    "proposition_type": pv.proposition_kind or "Desconhecido",
                    "mu": pv.mu,
                    "lambda": pv.lam,
                    "certainty": pv.certainty,
                    "contradiction": pv.contradiction,
                    "truth_value": pv.truth_value,
                    "state": pv.state,
                }
                for pv in pv_list
            ],
            many_valued_routes=[
                {
                    "left": left.proposition,
                    "left_type": left.proposition_kind or "Desconhecido",
                    "right": right.proposition,
                    "right_type": right.proposition_kind or "Desconhecido",
                    "route": route,
                    "confidence": confidence,
                    "explanation": explanation,
                }
                for left, right, route, confidence, explanation in self.L3.route_contradictions(pv_list)
            ],
            bert_classifications=[
                {
                    "proposition": judgment.proposicao,
                    "priority": judgment.prioridade,
                    "truth": judgment.epistemic_classification.truth,
                    "indeterminacy": judgment.epistemic_classification.indeterminacy,
                    "falsity": judgment.epistemic_classification.falsity,
                    "classification": judgment.epistemic_classification.classification,
                }
                for judgment, _ in filtered[:8]
            ],
            application_context=LogicLMSymbolicSolver.summarize_application_context(concepts),
        )

        self._log("\n[ETAPA 7] L4 — Síntese Russelliana")
        l2_priorities = {j.proposicao[:40]: j.prioridade for j, _ in filtered}
        result: SynthesisResult = self.L4.synthesize(pv_list, l2_priorities, prompt)
        l4_result = result
        l5_text = result.response

        # Contexto do agente (busca web/local) se ativo
        agent_context = ""
        if use_agent and run_search_for_context:
            try:
                agent_context = run_search_for_context(prompt)
                if agent_context and self.verbose:
                    self._log("\n[AGENTE] Contexto de busca obtido.")
            except Exception:
                pass

        # L4 — nota de fontes e auditoria (agora com contexto RAG disponível)
        l4_sources_note = self._build_citation_note(concepts, agent_context)
        l4_result.response = self._append_audit_block(
            l4_result.response,
            "L4",
            f"truth={l4_result.truth_value:.4f} certainty={l4_result.certainty:+.4f} contradiction={l4_result.contradiction:+.4f} state={l4_result.state} | {l4_sources_note}"
        )
        result.response = l4_result.response
        l5_text = result.response

        # L5 — Geração de resposta em texto livre
        gen_cfg = self._config.get("generation", {})
        final_cfg = self._config.get("finalization", {})
        l7_cfg = self._config.get("l7", {})

        base_provider = gen_cfg.get("provider", "ollama")
        if base_provider not in {"ollama", "custom_lm", "template"}:
            base_provider = "ollama"

        if final_cfg.get("provider") and final_cfg.get("provider") != base_provider:
            self._log(
                f"[PIPELINE] Ignorando provider de finalização '{final_cfg.get('provider')}' para usar provider base '{base_provider}'."
            )
        if l7_cfg.get("provider") and l7_cfg.get("provider") != base_provider:
            self._log(
                f"[PIPELINE] Ignorando provider L7 '{l7_cfg.get('provider')}' para usar provider base '{base_provider}'."
            )

        provider = base_provider
        if not skip_l5 and l5_generate and provider != "template":
            final_response = l5_generate(
                prompt,
                result,
                provider=provider,
                concepts_summary=concepts_summary,
                top_judgments=top_judgments,
                custom_lm_path=gen_cfg.get("custom_lm_path", ""),
                ollama_model=gen_cfg.get("ollama_model", "doninha8:latest"),
                ollama_host=gen_cfg.get("ollama_host", "http://localhost:11434"),
            )
            if agent_context and final_response:
                final_response = final_response + "\n\n[Contexto da busca]\n" + agent_context[:800]
            elif agent_context:
                final_response = result.response + "\n\n[Contexto da busca]\n" + agent_context[:800]
            else:
                final_response = final_response or result.response
            result = SynthesisResult(
                response=self._append_audit_block(
                    final_response,
                    "L5",
                    f"provider={provider} model={gen_cfg.get('ollama_model', 'doninha8:latest')} | audit=L1-L5: {self._summarize_judgments(judgments)}"
                ),
                truth_value=result.truth_value,
                certainty=result.certainty,
                contradiction=result.contradiction,
                state=result.state,
                supporting_evidence=result.supporting_evidence,
                falsified_hypotheses=result.falsified_hypotheses,
                confidence_label=result.confidence_label,
            )
        elif agent_context and result.response:
            result = SynthesisResult(
                response=self._append_audit_block(
                    result.response + "\n\n[Contexto da busca]\n" + agent_context[:800],
                    "L5",
                    f"provider={provider} model={gen_cfg.get('ollama_model', 'doninha8:latest')} | audit=L1-L5: {self._summarize_judgments(judgments)}"
                ),
                truth_value=result.truth_value,
                certainty=result.certainty,
                contradiction=result.contradiction,
                state=result.state,
                supporting_evidence=result.supporting_evidence,
                falsified_hypotheses=result.falsified_hypotheses,
                confidence_label=result.confidence_label,
            )

        l5_text = result.response

        if not skip_l6:
            final_text = self.L6.finalize_response(
                prompt=prompt,
                synthesis_result=result,
                epistemic_context=epistemic_context,
                generated_text=result.response,
                concepts_summary=concepts_summary,
                top_judgments=top_judgments,
                agent_context=agent_context,
            )
            final_text = self.L6.rewrite_response(
                prompt=prompt,
                synthesis_result=result,
                epistemic_context=epistemic_context,
                generated_text=final_text,
                concepts_summary=concepts_summary,
                top_judgments=top_judgments,
                agent_context=agent_context,
                provider=base_provider,
                custom_lm_path=final_cfg.get("custom_lm_path", gen_cfg.get("custom_lm_path", "")),
                ollama_model=final_cfg.get("ollama_model", gen_cfg.get("ollama_model", "doninha8:latest")),
                ollama_host=final_cfg.get("ollama_host", gen_cfg.get("ollama_host", "http://localhost:11434")),
            )
            result = SynthesisResult(
                response=self._append_audit_block(
                    final_text,
                    "L6",
                    f"provider={base_provider} model={final_cfg.get('ollama_model', gen_cfg.get('ollama_model', 'doninha8:latest'))} | epistemic={self._summarize_paraconsistent(pv_list)}"
                ),
                truth_value=result.truth_value,
                certainty=result.certainty,
                contradiction=result.contradiction,
                state=result.state,
                supporting_evidence=result.supporting_evidence,
                falsified_hypotheses=result.falsified_hypotheses,
                confidence_label=result.confidence_label,
            )

        l3_summary = ""
        if epistemic_context is not None and epistemic_context.proposition_states:
            top_states = epistemic_context.proposition_states[:3]
            l3_summary = "; ".join(
                f"{item.get('proposition', 'desconhecida')} → {item.get('state', 'n/a')} ({item.get('truth_value', 0):.2f})"
                for item in top_states
            )
            if epistemic_context.many_valued_routes:
                l3_summary += f"; rotas paraconsistentes: {len(epistemic_context.many_valued_routes)}"

        l7_cfg = self._config.get("l7", {})
        # Coletar alertas de incompatibilidade canônica gerados durante L1
        canonical_alerts = LogicLMSymbolicSolver.get_canonical_alerts() if LogicLMSymbolicSolver else []
        
        # === L7 — Texto Final Definitivo (Automático e Integrado) ===
        # Usa o agente de síntese final quando disponível, com fallback para a engine L7 original.
        if synthesize_final_agent is not None:
            final_text_l7 = synthesize_final_agent(
                prompt=prompt,
                l1_summary=concepts_summary,
                l2_summary=top_judgments,
                l3_summary=l3_summary,
                l4_response=l4_result.response,
                l5_text=l5_text,
                l6_text=result.response,
                provider=base_provider,
                model=l7_cfg.get("model", gen_cfg.get("ollama_model", "doninha8:latest")),
                temperature=l7_cfg.get("temperature", 0.7),
                max_tokens=l7_cfg.get("max_tokens", 4096),
            )
        else:
            final_text_l7 = self.L7.finalize_text(
                prompt=prompt,
                l1_summary=concepts_summary,
                l2_summary=top_judgments,
                l3_summary=l3_summary,
                l4_response=l4_result.response,
                l5_text=l5_text,
                l6_text=result.response,
                synthesis_result=l4_result,
                provider=base_provider,
                model=l7_cfg.get("model", gen_cfg.get("ollama_model", "doninha8:latest")),  # Padrão para ollama
                custom_lm_path=l7_cfg.get("custom_lm_path", gen_cfg.get("custom_lm_path", "")),
                canonical_alerts=canonical_alerts,
                temperature=l7_cfg.get("temperature", 0.7),
                max_tokens=l7_cfg.get("max_tokens", 4096),
            )
        result = SynthesisResult(
            response=self._append_audit_block(
                final_text_l7,
                "L7",
                f"provider={base_provider} model={l7_cfg.get('model', gen_cfg.get('ollama_model', 'doninha8:latest'))} | sources={'; '.join(self._collect_canonical_sources(concepts)[:6]) or 'nenhuma fonte local'} | L2={self._summarize_judgments(judgments)} | L3={self._summarize_paraconsistent(pv_list)}"
            ),
            truth_value=result.truth_value,
            certainty=result.certainty,
            contradiction=result.contradiction,
            state=result.state,
            supporting_evidence=result.supporting_evidence,
            falsified_hypotheses=result.falsified_hypotheses,
            confidence_label=result.confidence_label,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        self._log(f"\n[ETAPA 10] L7 — Texto Final Definitivo  ({elapsed:.1f} ms)\n")
        self._log(str(result))
        return result

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def repl(self) -> None:
        print("\n" + "═" * 60)
        print("  MODELO HÍBRIDO DE LLM — Fonseca")
        print("  Digite 'sair' para encerrar")
        print("═" * 60)
        while True:
            try:
                prompt = input("\nPrompt › ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                continue
            if prompt.lower() in {"sair", "exit", "quit"}:
                break
            self.process(prompt)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Modelo Híbrido de LLM — Pipeline L1–L7")
    parser.add_argument("--prompt", "-p", type=str, help="Pergunta única (imprime só a resposta)")
    parser.add_argument("--repl", action="store_true", help="Modo interativo")
    parser.add_argument("--demo", action="store_true", help="Rodar demonstração com prompts fixos")
    parser.add_argument("--config", type=str, help="Caminho para config.yaml")
    args, _ = parser.parse_known_args()

    config = load_config(Path(args.config)) if load_config and args.config else (load_config() if load_config else {})
    pipeline = HybridLLMPipeline(config=config, verbose=not args.prompt)

    if args.prompt:
        r = pipeline.process(args.prompt)
        print(r.response)
        return
    if args.repl:
        pipeline.repl()
        return
    if args.demo:
        for p in ["A água a 35 graus está quente ou fria?", "O que é a verdade?"]:
            pipeline.process(p)
            print()
        return
    # Default: demo + repl se --repl no argv antigo
    if "--repl" in sys.argv:
        pipeline.repl()
        return
    for p in ["A água a 35 graus está quente ou fria?", "O que é a verdade?"]:
        pipeline.process(p)
        print()


if __name__ == "__main__":
    main()

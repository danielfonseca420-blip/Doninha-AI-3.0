"""
CAMADA L7 — Texto Final Definitivo (Automático e Robusto)
==========================================================
Gera o texto final de alta qualidade a partir do raciocínio acumulado
nas camadas L1 a L6.

A camada L7 funciona como um prompt adicional de escrita: ela recebe os
sumários das camadas anteriores e transforma o conteúdo em um único
bloco contínuo, fluido e persuasivo.

Suporta múltiplos providers:
- ollama: para modelos rodando localmente
- custom_lm: para modelos customizados
- template: fallback que retorna texto sem LLM
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import logging
from l4_synthesis import SynthesisResult
from layer_titles import LAYER_TITLES

logger = logging.getLogger(__name__)

try:
    from l5_generation import generate_with_custom_lm
except Exception:
    generate_with_custom_lm = None  # type: ignore

try:
    import ollama
except Exception:
    ollama = None  # type: ignore


class FinalTextEngine:
    """Gera o texto final definitivo a partir do raciocínio L1–L6."""

    # Classificação de audiência
    AUDIENCE_PROFILES = {
        "leigo": {
            "description": "Público geral sem conhecimento técnico especializado",
            "style": "Linguagem simples e acessível, analogias concretas do dia a dia, evitar notação formal, foco em aplicações práticas e conclusões úteis",
            "examples": ["o que é", "como funciona", "explicar", "simples"]
        },
        "técnico": {
            "description": "Profissional da área com conhecimento técnico intermediário",
            "style": "Usar terminologia específica da área, incluir referências conceituais, evitar tabelas de estados complexas, manter rigor técnico sem excesso de formalismo",
            "examples": ["análise", "implementação", "método", "técnica", "profissional"]
        },
        "acadêmico": {
            "description": "Pesquisador ou acadêmico com formação avançada",
            "style": "Notação completa e formal, referências bibliográficas detalhadas, incluir modo debug/disponibilidade de estados internos, rigor acadêmico completo",
            "examples": ["teoria", "formal", "demonstração", "referência", "acadêmico", "pesquisa"]
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa a FinalTextEngine com configurações opcionais.
        
        Args:
            config: Dicionário com configurações de L7, incluindo:
                - provider: 'ollama', 'custom_lm', ou 'template'
                - model: Nome do modelo (para ollama)
                - temperature: Temperatura de geração (padrão: 0.7)
                - max_tokens: Número máximo de tokens (padrão: 4096)
                - custom_lm_path: Caminho do modelo customizado (para custom_lm)
        """
        self.config = config or {}
        self.l7_config = self.config.get("l7", {})
        
    def _build_l7_prompt(self, 
                        prompt: str,
                        l1_summary: str,
                        l2_summary: str,
                        l3_summary: str,
                        l4_response: str,
                        l5_text: str,
                        l6_text: str,
                        audience_profile: str = "técnico",
                        full_synthesis: Optional[str] = None) -> str:
        """
        Constrói automaticamente o prompt L7 para geração de texto final.
        
        Este método agrega todo o raciocínio das camadas L1-L6 e produz
        um prompt bem estruturado e adaptado ao perfil de audiência.
        
        Args:
            prompt: Pergunta/prompt original do usuário
            l1_summary: Resumo de conceitos extraídos (L1)
            l2_summary: Resumo de juízos kantianos (L2)
            l3_summary: Resumo de análise paraconsistente (L3)
            l4_response: Resposta da síntese russelliana (L4)
            l5_text: Texto gerado em L5 (se disponível)
            l6_text: Texto refinado de L6
            audience_profile: Perfil de audiência ('leigo', 'técnico', 'acadêmico')
            full_synthesis: Síntese completa opcional
            
        Returns:
            String com prompt bem estruturado para geração automática
        """
        lines = []
        
        # SEÇÃO 1: Instrução Base
        lines.append("Você é um excelente escritor técnico e comunicador, especializado em sintetizar raciocínios complexos em textos claros, profundos e agradáveis de ler.")
        lines.append("")
        lines.append("Sua função é gerar o TEXTO FINAL DEFINITIVO a partir de todo o raciocínio desenvolvido nas camadas L1 a L6.")
        lines.append("")
        
        # SEÇÃO 2: Contexto do Prompt Original
        lines.append("═" * 70)
        lines.append("PROMPT ORIGINAL DO USUÁRIO:")
        lines.append("═" * 70)
        lines.append(prompt)
        lines.append("")
        
        # SEÇÃO 3: Resumo das Camadas
        lines.append("═" * 70)
        lines.append("RACIOCÍNIO ACUMULADO (CAMADAS L1–L6):")
        lines.append("═" * 70)
        lines.append(f"L1 - Conceitos Extraídos: {l1_summary or 'Não disponível'}")
        lines.append(f"L2 - Juízos Kantianos: {l2_summary or 'Não disponível'}")
        lines.append(f"L3 - Análise Paraconsistente: {l3_summary or 'Não disponível'}")
        lines.append(f"L4 - Síntese Russelliana: {l4_response or 'Não disponível'}")
        lines.append(f"L5 - Geração de Resposta: {l5_text or 'Não disponível'}")
        lines.append(f"L6 - Refinamento Final: {l6_text or 'Não disponível'}")
        lines.append("")
        
        # SEÇÃO 4: Perfil de Audiência
        profile_data = self.AUDIENCE_PROFILES.get(audience_profile, self.AUDIENCE_PROFILES["técnico"])
        lines.append("═" * 70)
        lines.append(f"PERFIL DE AUDIÊNCIA: {audience_profile.upper()}")
        lines.append("═" * 70)
        lines.append(f"Descrição: {profile_data['description']}")
        lines.append(f"Estilo recomendado: {profile_data['style']}")
        lines.append("")
        
        # SEÇÃO 5: Diretivas de Formatação (OBRIGATÓRIAS)
        lines.append("═" * 70)
        lines.append("DIRETIVAS DE FORMATAÇÃO (OBRIGATÓRIAS):")
        lines.append("═" * 70)
        lines.append("• Formato: Texto fluido, com parágrafos quando necessário, sem títulos, subtítulos, bullets ou numeração.")
        lines.append("• Abertura: Comece diretamente com a tese ou resposta principal (1-2 frases fortes e claras).")
        lines.append("• Estrutura: Desenvolvimento gradual das premissas, nuances e evolução do pensamento.")
        lines.append("• Integração: Harmonize todas as camadas de forma natural, mostrando a evolução do raciocínio.")
        lines.append("• Tone: Profissional, confiante e acessível. Explique termos técnicos quando necessários.")
        lines.append("• Variação: Use frases de tamanhos variados com transições naturais e sofisticadas.")
        lines.append("• Ênfase: Destaque ideias importantes via posicionamento e repetição sutil (não óbvia).")
        lines.append("• Rigor: Inclua tensões, trade-offs e incertezas com elegância e maturidade intelectual.")
        lines.append("")
        
        # SEÇÃO 6: Tarefa Final
        lines.append("═" * 70)
        lines.append("TAREFA:")
        lines.append("═" * 70)
        lines.append("Transforme todo esse raciocínio em uma DISSERTAÇÃO EXPOSITIVA FLUIDA, COESA E NATURAL.")
        lines.append("Escreva o texto final agora:")
        lines.append("═" * 70)
        lines.append("")
        
        return "\n".join(lines)

    @classmethod
    def _classify_audience(cls, prompt: str, l1_summary: str, l2_summary: str, l3_summary: str) -> str:
        """
        Classifica o perfil da audiência baseado no prompt e contexto das camadas.
        Retorna: 'leigo', 'técnico', ou 'acadêmico'
        """
        # Combinar todo o contexto para análise
        full_context = f"{prompt} {l1_summary} {l2_summary} {l3_summary}".lower()

        # Contar termos técnicos e indicadores de nível
        technical_indicators = {
            "leigo": 0,
            "técnico": 0,
            "acadêmico": 0
        }

        # Análise de vocabulário e termos
        for profile, data in cls.AUDIENCE_PROFILES.items():
            for keyword in data["examples"]:
                if keyword.lower() in full_context:
                    technical_indicators[profile] += 1

        # Análise de extensão e complexidade
        prompt_length = len(prompt.split())
        has_formal_terms = any(term in full_context for term in [
            "formal", "demonstração", "teorema", "axioma", "paradigma",
            "epistemologia", "ontologia", "metafísica", "transcendental"
        ])
        has_technical_jargon = any(term in full_context for term in [
            "lógica paraconsistente", "juízo kantiano", "síntese russelliana",
            "valor de verdade", "contradição", "cognição"
        ])

        # Regras de classificação
        if has_formal_terms or prompt_length > 50 or "referência" in full_context:
            return "acadêmico"
        elif has_technical_jargon or technical_indicators["técnico"] > technical_indicators["leigo"]:
            return "técnico"
        elif technical_indicators["leigo"] > 0 or prompt_length < 20:
            return "leigo"
        else:
            # Padrão: analisar padrão da pergunta
            question_patterns = {
                "acadêmico": ["por que", "como se explica", "qual a teoria", "demonstre"],
                "técnico": ["como implementar", "qual método", "análise de", "técnica para"],
                "leigo": ["o que é", "para que serve", "como funciona", "exemplo"]
            }

            for profile, patterns in question_patterns.items():
                if any(pattern in prompt.lower() for pattern in patterns):
                    return profile

            return "técnico"  # padrão seguro

    def _enhance_with_writer_prompt(
        self,
        base_text: str,
        prompt: str,
        audience_profile: str,
        synthesis_result: Optional[SynthesisResult] = None,
        canonical_alerts: Optional[list] = None
    ) -> str:
        """
        Aprimora o texto base usando o prompt de redação (fallback sem LLM).
        Usada quando nenhum provider está disponível.
        """
        # Aqui poderíamos aplicar algumas transformações/enhancements
        # que não requerem LLM, como formatting, reorganização, etc.
        return self._build_writer_prompt(
            prompt=prompt,
            l1_summary="",
            l2_summary="",
            l3_summary="",
            l4_response="",
            l5_text="",
            l6_text=base_text,
            synthesis_result=synthesis_result,
            canonical_alerts=canonical_alerts,
            audience_profile=audience_profile
        )


    def finalize_text(
        self,
        prompt: str,
        l1_summary: str = "",
        l2_summary: str = "",
        l3_summary: str = "",
        l4_response: str = "",
        l5_text: str = "",
        l6_text: str = "",
        synthesis_result: Optional[SynthesisResult] = None,
        provider: str = "ollama",
        model: str = "doninha8:latest",
        custom_lm_path: str = "",
        canonical_alerts: Optional[list] = None,
        audience_profile: Optional[str] = None,
        **kwargs) -> str:
        """
        Gera o texto final definitivo de forma automática e robusta.
        
        Suporta múltiplos providers:
        - ollama: Executa modelos locais via Ollama
        - custom_lm: Usa modelo LM customizado
        - template: Retorna o melhor resultado de L6 sem LLM (fallback)
        
        Args:
            prompt: Pergunta/prompt original do usuário
            l1_summary: Resumo de conceitos (L1)
            l2_summary: Resumo de juízos kantianos (L2)
            l3_summary: Resumo de análise paraconsistente (L3)
            l4_response: Resposta da síntese (L4)
            l5_text: Texto gerado (L5)
            l6_text: Texto refinado (L6)
            synthesis_result: Resultado da síntese L4
            provider: 'ollama', 'custom_lm', ou 'template'
            model: Nome do modelo Ollama
            custom_lm_path: Caminho do modelo customizado
            canonical_alerts: Alertas de incompatibilidade semântica
            audience_profile: 'leigo', 'técnico', ou 'acadêmico'
            **kwargs: Argumentos adicionais (temperature, max_tokens, etc.)
            
        Returns:
            String com o texto final gerado
        """
        
        # 1. Classificar audiência se não foi fornecida
        if audience_profile is None:
            audience_profile = self._classify_audience(prompt, l1_summary, l2_summary, l3_summary)
        
        # 2. Construir prompt L7 automático
        l7_prompt = self._build_l7_prompt(
            prompt=prompt,
            l1_summary=l1_summary,
            l2_summary=l2_summary,
            l3_summary=l3_summary,
            l4_response=l4_response,
            l5_text=l5_text,
            l6_text=l6_text,
            audience_profile=audience_profile,
            full_synthesis=synthesis_result.response if synthesis_result else None
        )
        
        # 3. Gerar texto usando o provider selecionado
        generated_text = None
        
        if provider == "ollama" and ollama:
            generated_text = self._generate_with_ollama(
                prompt=l7_prompt,
                model=self.l7_config.get("model", model),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4096)
            )
        elif provider == "custom_lm" and generate_with_custom_lm:
            custom_path = custom_lm_path or self.l7_config.get("custom_lm_path", "")
            if custom_path:
                generated_text = self._generate_with_custom_lm(
                    prompt=l7_prompt,
                    model_path=custom_path
                )

        if provider == "template":
            return (l6_text or l5_text or l4_response or "").strip()

        if generated_text:
            return generated_text.strip()

        return (l6_text or l5_text or l4_response or "").strip()

    def _generate_with_ollama(self, prompt: str, model: str, temperature: float = 0.7, max_tokens: int = 4096) -> Optional[str]:
        """
        Gera texto usando Ollama (modelos locais).
        
        Args:
            prompt: Prompt para geração
            model: Nome do modelo (e.g., 'llama2', 'neural-chat', 'mistral')
            temperature: Controla criatividade (0.0-1.0)
            max_tokens: Limite de tokens de saída
            
        Returns:
            Texto gerado ou None se falhar
        """
        try:
            if not ollama:
                logger.error("Ollama não está instalado")
                return None
            
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 8192
                }
            )
            
            generated = response.get("message", {}).get("content", "").strip()
            if generated:
                logger.info(f"L7 (ollama/{model}): Texto gerado com sucesso ({len(generated)} chars)")
                return generated
            else:
                logger.warning("Ollama retornou resposta vazia")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao usar Ollama: {e}")
            return None

    def _generate_with_custom_lm(self, prompt: str, model_path: str) -> Optional[str]:
        """
        Gera texto usando modelo LM customizado.
        
        Args:
            prompt: Prompt para geração
            model_path: Caminho do modelo customizado
            
        Returns:
            Texto gerado ou None se falhar
        """
        try:
            if not generate_with_custom_lm:
                logger.error("generate_with_custom_lm não está disponível")
                return None
            
            generated = generate_with_custom_lm(prompt, model_path)
            if generated:
                logger.info(f"L7 (custom_lm): Texto gerado com sucesso ({len(generated)} chars)")
                return generated
            else:
                logger.warning("Custom LM retornou resposta vazia")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao usar Custom LM: {e}")
            return None



    def _build_writer_prompt(
        self,
        prompt: str,
        l1_summary: str,
        l2_summary: str,
        l3_summary: str,
        l4_response: str,
        l5_text: str,
        l6_text: str,
        synthesis_result: Optional[SynthesisResult] = None,
        canonical_alerts: Optional[list] = None,
        audience_profile: str = "técnico",
    ) -> str:
        lines = [
            "Você é um excelente escritor técnico e comunicador, com capacidade de sintetizar raciocínios complexos em textos claros e persuasivos.",
            "Sua função é gerar o texto final de alta qualidade a partir do raciocínio desenvolvido nas camadas L1 a L6.",
            "Tarefa: transforme todo o raciocínio acumulado nas camadas L1 a L6 em uma dissertação expositiva fluida, coesa e natural, usando parágrafos claros quando for apropriado.",
            "Público-alvo: leitor inteligente de nível intermediário (não é especialista no tema).",
            "Formato: texto fluido, com parágrafos quando necessário, sem títulos, subtítulos, bullets ou qualquer marcação.",
            "Estrutura recomendada: comece diretamente com a tese ou resposta principal em 1-2 frases fortes e claras. Em seguida, desenvolva as premissas, nuances e evoluções do pensamento.",
            "Integre harmoniosamente o conteúdo das camadas anteriores, mostrando a evolução natural do raciocínio e destacando tensões, trade-offs e incertezas com elegância.",
            "Linguagem: clara, conversacional e precisa. Use termos técnicos quando necessários, explicando-os na sequência.",
            "Estilo: profissional, acessível, rigoroso e fácil de ler.",
            "",
            f"PERFIL DA AUDIÊNCIA CLASSIFICADO: {audience_profile.upper()}",
        ]

        # Adicionar instruções específicas do perfil
        profile_data = self.AUDIENCE_PROFILES.get(audience_profile, self.AUDIENCE_PROFILES["técnico"])
        lines.append(f"Descrição do perfil: {profile_data['description']}")
        lines.append(f"Instruções de estilo específicas: {profile_data['style']}")
        lines.append("")
        lines.append(f"Pergunta do usuário: {prompt}")
        lines.append("")
        lines.append("Raciocínio acumulado L1–L6:")
        lines.append(f"L1 - {LAYER_TITLES['l1']}: {l1_summary or 'Não disponível.'}")
        lines.append(f"L2 - {LAYER_TITLES['l2']}: {l2_summary or 'Não disponível.'}")
        lines.append(f"L3 - {LAYER_TITLES['l3']}: {l3_summary or 'Não disponível.'}")
        lines.append(f"L4 - {LAYER_TITLES['l4']}: {l4_response or 'Não disponível.'}")
        lines.append(f"L5 - {LAYER_TITLES['l5']}: {l5_text or 'Não disponível.'}")
        lines.append(f"L6 - {LAYER_TITLES['l6']}: {l6_text or 'Não disponível.'}")
        lines.append(f"L7 - {LAYER_TITLES['l7']}: texto final de síntese e redação.")
        lines.append("")

        # Adicionar informações da síntese L4 se disponível
        if synthesis_result:
            lines.append(f"Estado da síntese L4: {synthesis_result.state}")
            lines.append(f"Valor de verdade L4: {synthesis_result.truth_value:.2f}")
            lines.append(f"Certeza L4: {synthesis_result.certainty:+.2f}")
            lines.append(f"Contradição L4: {synthesis_result.contradiction:+.2f}")
            lines.append("")
        else:
            lines.append("Estado da síntese L4: Não disponível.")
            lines.append("")

        # Adicionar alertas de incompatibilidade canônica
        if canonical_alerts:
            lines.append("Alertas de incompatibilidade canônica:")
            for alert in canonical_alerts:
                lines.append(f"- Conceito '{alert['concept']}': {alert['canonical_context']}")
                lines.append(f"  Uso incompatível detectado: {alert['incompatible_usage']}")
            lines.append("")
            lines.append("IMPORTANTE: Inclua ressalvas no texto final sobre estes usos incompatíveis dos conceitos.")
            lines.append("")

        lines.append("Escreva o texto final agora.")

        return "\n".join(lines)

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.split()).strip()

    def _ensure_single_paragraph(self, text: str) -> str:
        return " ".join(text.replace("\n", " ").split()).strip()

# Doninha IA — Middleware Neuro-Simbólico Híbrido

> Doninha transforma qualquer LLM base (Ollama, template, custom LM) em um modelo híbrido neuro-simbólico com raciocínio estruturado, verificação epistemológica e auditoria explícita.

## O que é o Doninha?

O Doninha não é apenas um chamador de LLM. Ele é um middleware de raciocínio em 7 camadas:

1. extrai conceitos e relações semânticas;
2. constrói juízos e prioridades epistemológicas;
3. avalia proposições com lógica paraconsistente;
4. sintetiza com base russelliana e verificação CoVe;
5. gera texto fluido com o LLM base;
6. refina a resposta final para clareza e consistência;
7. produz uma síntese definitiva, auditável e pronta para uso em produção.

Em outras palavras, o Doninha converte um LLM generativo comum em um modelo híbrido que mistura:
- geração estatística do modelo base;
- estrutura simbólica (conceitos, juízos, regras, correspondência);
- verificação epistemológica e auditoria de confiança.

---

## Arquitetura resumida

```text
Prompt
  ├─ L1: Tábua de Conceitos
  ├─ L2: Juízos Kantianos
  ├─ L3: Lógica Paraconsistente
  ├─ L4: Síntese Russelliana + CoVe
  ├─ L5: Geração textual (LLM base)
  ├─ L6: Refinamento final
  └─ L7: Texto definitivo e auditável
```

---

## Ferramentas e módulos usados no fluxo

| Camada | Função principal | Ferramentas / motores usados |
|---|---|---|
| L1 | Extrair conceitos e relações semânticas | `ConceptTable`, `LogicLMSymbolicSolver`, `knowledge_base.py` |
| L2 | Priorizar proposições e juízos | `KantianJudgmentEngine` |
| L3 | Avaliar μ/λ, estado, certeza e contradição | `ParaconsistentEngine`, `ManyValuedRouter` |
| L4 | Síntese russelliana + verificação | `RussellianSynthesisEngine`, `ChainOfVerificationAgent` |
| L5 | Geração em texto livre | `l5_generation.py`, Ollama / custom LM / fallback template |
| L6 | Refinamento final e escrita fluida | `FinalResponseEngine`, `l6_final_response.py` |
| L7 | Texto final definitivo e síntese integrada | `FinalTextEngine`, `agente_sintese_final.py` |

---

## Passo a passo das camadas L1 a L7

### L1 — Tábua de Conceitos
A camada L1 identifica termos, definições, relações semânticas e contexto de aplicação.

O que ela faz:
- extrai conceitos do prompt;
- mapeia sinonímia, antonímia, hiperonímia, hiponímia e relações de domínio;
- usa a base de conhecimento local e, quando disponível, o contexto RAG.

Por que isso importa:
- evita que o modelo responda como se todos os termos fossem equivalentes;
- cria a base semântica para o restante do pipeline.

### L2 — Juízos Kantianos
A camada L2 transforma o material da L1 em proposições prioritárias.

O que ela faz:
- constrói juízos a partir de conceitos extraídos;
- atribui prioridade e classificação epistemológica;
- separa hipóteses mais fortes, intermediárias e fracas.

Por que isso importa:
- o modelo deixa de responder apenas “em linguagem livre” e passa a operar com um esqueleto argumentativo.

### L3 — Lógica Paraconsistente
A camada L3 avalia cada proposição com métricas de evidência:
- μ: evidência favorável;
- λ: evidência contrária;
- certainty: grau de certeza;
- contradiction: grau de contradição;
- truth_value: valor-verdade normalizado;
- state: verdadeiro, falso, indeterminado, inconsistente local ou intermediário.

O que ela faz:
- usa a lógica paraconsistente para não colapsar a resposta em “verdadeiro/falso” bruto;
- preserva ambiguidade e contradições locais quando o material é complexo.

### L4 — Síntese Russelliana + CoVe
Aqui a síntese final do raciocínio começa a tomar forma.

O que ela faz:
- combina L2 + L3 em um argumento mais robusto;
- usa a teoria da correspondência e da equivalência (Russell) como princípio de alinhamento com fatos e evidências;
- aplica Chain of Verification (CoVe) para revisar e marcar limitações da resposta.

Por que isso importa:
- esta camada é o núcleo do modelo híbrido: a resposta deixa de ser apenas gerada e passa a ser fundamentada.

### L5 — Geração textual via LLM base
A camada L5 transforma a síntese estruturada em texto natural, usando:
- Ollama (`gpt-oss:120b-cloud` por padrão);
- modelo customizado (`custom_lm`), se configurado;
- fallback para a própria resposta L4 quando o gerador não estiver disponível.

Por que isso importa:
- a geração textual ganha rigor epistemológico, porque o LLM não começa do zero: ele recebe contexto de L1–L4.

### L6 — Refinamento final
A camada L6 polia o texto gerado.

O que ela faz:
- melhora clareza, fluidez e coesão;
- mantém o tom técnico e acessível;
- preserva as métricas epistemológicas e destaca incertezas quando existirem.

### L7 — Síntese final definitiva
A camada L7 entrega a versão final, pronta para interface, API ou agente.

O que ela faz:
- integra L1–L6 em uma resposta final coesa;
- usa `FinalTextEngine` e, quando disponível, `agente_sintese_final.py`;
- acrescenta rastro de auditoria e contexto de verificação.

---

## Verificabilidade e auditoria

O Doninha foi projetado para ser auditável, não apenas fluente.

### O que é verificado
- estado lógico da proposição (μ, λ, truth_value, certainty, contradiction);
- priorização epistemológica das hipóteses (L2);
- consistência estrutural das respostas (L4 via CoVe);
- presença ou ausência de fontes bibliográficas/canonicais;
- uso de contexto RAG/agent quando o usuário opta por busca externa.

### Como a auditoria aparece na saída
A pipeline gera blocos como:
- `[AUDIT L4]`
- `[AUDIT L7]`

Esses blocos registram:
- provider/modelo usado;
- estado lógico da síntese;
- métrica de verdade e certeza;
- nota de fontes bibliográficas ou aviso explícito de ausência de fontes verificáveis.

### Regras de segurança epistemológica
- se houver fonte bibliográfica canônica disponível, a resposta deve mencionar isso;
- se não houver verificação externa confiável, a resposta deve explicitar esse limite em linguagem clara;
- a resposta final é sempre tratada como uma síntese com evidência e incerteza, não como uma verdade absoluta.

---

## Como usar

### 1) Preparar ambiente

```bash
python -m pip install -r requirements.txt
```

### 2) Iniciar Ollama

```bash
ollama serve
ollama pull gpt-oss:120b-cloud
ollama create -f Modelfile doninha8
```

### 3) Executar a pipeline principal

```bash
python pipeline.py --demo
python pipeline.py --prompt "Explique lógica paraconsistente em 5 linhas"
python pipeline.py --repl
```

### 4) Executar o middleware Chainlit

```bash
python app.py
```

### 5) Executar a API REST

```bash
python api.py
```

Endpoints principais:
- `POST /process`
- `POST /chat`
- `GET /health`
- `POST /agent`

---

## Configuração principal

O arquivo `config.yaml` controla:
- base de conhecimento (`knowledge_base`);
- L3, L4, L5, L6 e L7;
- modelo base Ollama (`gpt-oss:120b-cloud`);
- host/porta da API;
- agente e contexto de chat.

Exemplo de configuração atual:

```yaml
generation:
  provider: "ollama"
  ollama_model: "gpt-oss:120b-cloud"
  ollama_host: "http://localhost:11434"

finalization:
  provider: "ollama"
  ollama_model: "gpt-oss:120b-cloud"

l7:
  provider: "ollama"
  model: "gpt-oss:120b-cloud"
```

---

## RAG híbrido

O projeto também oferece uma variante com RAG híbrido:

```bash
python pipeline_with_rag_integration.py --prompt "O que é epistemologia?" --strategy hybrid
python example_rag_hybrid_usage.py
```

---

## Observações finais

- O modelo base atual é `gpt-oss:120b-cloud`.
- A pipeline principal funciona com fallback de template quando o LLM base não está disponível.
- Para desenvolvimento local, Ollama continua sendo o caminho recomendado.
- O Doninha não substitui o LLM; ele o organiza, valida, enquadra e auditávelmente orienta para uma resposta mais robusta.

---

## Referência

Daniel Fonseca - Criador da Inteligencia Artificial Doninha, um middleware que transforma qualquer LLM em um modelo neuro simbólico híbrido com ferramentas de auditoria.
A IA Doninha tem sua fundamentação teórica presente no artigo "Uma verdadeira Epistemologia para a Inteligência Artificial"
Todos os direitos reservados à Daniel Barros Fonseca sujeitos a licenciamento para uso de terceiros.
A Tecnologia da IA Doninha encontra-se em processo corrente de registro de Patente. Plágios ou apropriações indevidas desta tecnologia proprietária estarão sujeitas à indenização, reparação de danos, reparação de danos por lucros perdidos e à obrigatoriedade de publicação de carta de retratação por espionagem industrial.

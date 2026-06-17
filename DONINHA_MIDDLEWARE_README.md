# DONINHA IA MIDDLEWARE — Guia de Uso

## Visão Geral

`doninha_middleware.py` é um **arquivo Python único e self-contained** que consolida todo o middleware neuro-simbólico Doninha em uma interface simples e poderosa.

**Estrutura**: 7 camadas epistemológicas + interface unificada
- **L1**: Tábua de Conceitos (Aristotélica)
- **L2**: Juízos Kantianos
- **L3**: Lógica Paraconsistente (μ/λ, 12 estados QUPC)
- **L4**: Síntese Russelliana + Chain of Verification
- **L5**: Geração textual (multi-LLM)
- **L6**: Refinamento final
- **L7**: Síntese definitiva com auditoria

## Instalação

### Pré-requisitos

```bash
pip install requests  # Para APIs
```

### Dependências Opcionais

```bash
# Para OpenAI
pip install openai

# Para Anthropic
pip install anthropic

# Para Google Gemini
pip install google-generativeai

# Para Ollama local
pip install ollama
```

### Uso Básico

```python
from doninha_middleware import DoninhaMiddleware

# Inicializa com fallback (sem API externa)
middleware = DoninhaMiddleware()

# Processa um prompt
response = middleware.process("O que é verdade?")
print(response)
```

## Exemplos de Uso

### 1. Modo Fallback (Sem APIs)

```python
from doninha_middleware import DoninhaMiddleware

middleware = DoninhaMiddleware({
    "provider": "fallback"
})

prompt = "Qual é a diferença entre conhecimento e crença?"
response = middleware.process(prompt)
print(response)
```

**Quando usar**: Prototipagem rápida, testes, ambiente sem internet

### 2. Com OpenAI GPT-4

```python
from doninha_middleware import DoninhaMiddleware

middleware = DoninhaMiddleware({
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "sk-...",  # Sua chave OpenAI
    "temperature": 0.3
})

response = middleware.process("Explique a epistemologia de Russell")
print(response)
```

### 3. Com Anthropic Claude

```python
middleware = DoninhaMiddleware({
    "provider": "anthropic",
    "model": "claude-3-opus-20240229",
    "api_key": "sk-ant-...",
    "temperature": 0.3
})

response = middleware.process("O que é lógica paraconsistente?")
```

### 4. Com Google Gemini

```python
middleware = DoninhaMiddleware({
    "provider": "gemini",
    "model": "gemini-pro",
    "api_key": "AIzaSy...",
    "temperature": 0.3
})

response = middleware.process("Defina silogismo aristotélico")
```

### 5. Com Ollama Local

```python
middleware = DoninhaMiddleware({
    "provider": "ollama",
    "model": "mistral:7b"  # ou seu modelo local
})

response = middleware.process("O que é verdade?")
```

### 6. Interface de Chat

```python
middleware = DoninhaMiddleware({
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "sk-..."
})

messages = [
    {"role": "user", "content": "Olá, como você funciona?"},
    {"role": "assistant", "content": "Sou um middleware epistemológico..."},
    {"role": "user", "content": "Qual é sua abordagem de verdade?"}
]

response = middleware.chat(messages)
print(response)
```

## API Completa

### Classe: `DoninhaMiddleware`

#### Inicialização

```python
DoninhaMiddleware(config: Optional[Dict[str, Any]] = None)
```

**Configuração disponível**:
```python
{
    "provider": "fallback" | "openai" | "anthropic" | "gemini" | "ollama",
    "model": "gpt-4" | "claude-3-opus-20240229" | "gemini-pro" | "mistral" | etc,
    "api_key": "sua-chave-de-api",
    "temperature": 0.3,  # [0.0, 1.0]
}
```

#### Métodos Principais

##### `process(prompt: str) -> str`

Processa um prompt através de todas as 7 camadas.

```python
response = middleware.process("Sua pergunta aqui")
```

**O que acontece**:
1. **L1**: Extrai conceitos semânticos
2. **L2**: Gera juízos kantianos
3. **L3**: Avalia paraconsistentemente (μ/λ)
4. **L4**: Sintetiza via correspondência russelliana
5. **L5**: Gera resposta com LLM
6. **L6**: Refina o texto
7. **L7**: Adiciona auditoria e metadados

##### `chat(messages: List[Dict[str, str]], system_prompt: str = "") -> str`

Interface compatível com chat APIs.

```python
messages = [
    {"role": "user", "content": "Pergunta 1"},
    {"role": "assistant", "content": "Resposta 1"},
    {"role": "user", "content": "Pergunta 2"}
]

response = middleware.chat(messages)
```

##### `set_config(config: Dict[str, Any]) -> None`

Atualiza configuração em tempo de execução.

```python
middleware.set_config({
    "provider": "openai",
    "model": "gpt-4"
})
```

##### `get_status() -> Dict[str, Any]`

Retorna status atual do middleware.

```python
status = middleware.get_status()
# {'provider': 'openai', 'model': 'gpt-4', 'status': 'operacional', ...}
```

## Estrutura de Resposta

Cada resposta inclui:

1. **Texto principal**: Resposta sintetizada
2. **Análise epistemológica**:
   - Valor de verdade (0-1)
   - Estado lógico (V, F, ⊥, T, etc)
   - Nível de confiança
3. **Bloco [AUDIT L7]**: Metadados auditáveis
   - Timestamp
   - Grau de certeza (Gc)
   - Grau de contradição (Gct)
   - Conceitos integrados
   - Juízos kantianos
   - Evidências e hipóteses descartadas

## Estados Lógicos (12)

O middleware suporta os 12 estados do reticulado paraconsistente:

| Estado | Descrição |
|--------|-----------|
| **V** | Verdadeiro (μ alto, λ baixo) |
| **F** | Falso (μ baixo, λ alto) |
| **T** | Inconsistente (μ alto, λ alto) |
| **⊥** | Indeterminado (μ baixo, λ baixo) |
| **QV** | Quase Verdadeiro |
| **QF** | Quase Falso |
| E 6 outros estados transitórios | |

## Exemplos de Saída

### Pergunta Epistemológica

**Input**:
```
"O que define a verdade segundo Russell?"
```

**Output (resumido)**:
```
Proposição central: A verdade é correspondência entre crença e fato.

Análise epistemológica:
- Valor de verdade: 0.82
- Estado: Verdadeiro
- Confiança: Alta Confiança

Resposta sintetizada:
Russell define verdade como a correspondência entre uma crença e 
os fatos da realidade. Uma proposição é verdadeira se há uma 
correspondência entre o que se acredita e o que existe no mundo.

[AUDIT L7 — SÍNTESE FINAL AUDITÁVEL]
Timestamp: 2026-06-12T09:14:16...
Estado lógico: Verdadeiro
Valor de verdade: 0.8200
Grau de certeza (Gc): +0.6400
Conceitos integrados (L1): verdade, correspondência, crença, fato
Juízos kantianos (L2): [Universal/Afirmativo/Categórico/Assertórico]
[FIM AUDIT L7]
```

## Casos de Uso

### 1. Chatbot Epistemológico

```python
middleware = DoninhaMiddleware({
    "provider": "openai",
    "model": "gpt-4",
    "api_key": os.getenv("OPENAI_API_KEY")
})

def chat_loop():
    while True:
        user_input = input("Você: ")
        if user_input.lower() in ["sair", "exit"]:
            break
        response = middleware.process(user_input)
        print(f"Doninha: {response}\n")

chat_loop()
```

### 2. API REST

```python
from flask import Flask, request, jsonify
from doninha_middleware import DoninhaMiddleware

app = Flask(__name__)
middleware = DoninhaMiddleware({"provider": "openai", "model": "gpt-4"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    prompt = data.get("prompt", "")
    response = middleware.process(prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5000)
```

### 3. Processamento em Lote

```python
questions = [
    "O que é conhecimento?",
    "Diferença entre verdade e realidade",
    "Explique a lógica paraconsistente"
]

middleware = DoninhaMiddleware({"provider": "fallback"})

results = [middleware.process(q) for q in questions]
for q, r in zip(questions, results):
    print(f"Q: {q}\nA: {r[:100]}...\n")
```

## Troubleshooting

### Erro: "Módulo openai não encontrado"

```bash
pip install openai
```

### Erro: "Invalid API Key"

Verifique sua chave de API:
```python
import os
os.getenv("OPENAI_API_KEY")  # Deve retornar sua chave
```

### Ollama não funciona

Certifique-se de que Ollama está rodando:
```bash
ollama serve
```

E que o modelo existe:
```bash
ollama list
```

## Performance

- **Fallback (sem LLM)**: ~100-200ms
- **Ollama local**: ~500ms-2s (depende do modelo)
- **OpenAI/Anthropic**: ~1-5s (depende da latência de rede)

## Arquitetura Técnica

```
Prompt
  ├─ L1: Extrai conceitos semânticos
  │   └─ Mapeia sinonímia, antonímia, hiponímia
  │
  ├─ L2: Gera juízos kantianos
  │   └─ Quantidade, Qualidade, Relação, Modalidade
  │
  ├─ L3: Avalia paraconsistentemente
  │   └─ Calcula μ, λ, Gc, Gct, estado lógico
  │
  ├─ L4: Sintetiza russelliana
  │   └─ Correspondência ↔ Fato, Chain of Verification
  │
  ├─ L5: Gera resposta
  │   └─ OpenAI | Anthropic | Gemini | Ollama | Fallback
  │
  ├─ L6: Refina texto
  │   └─ Normaliza, adiciona contexto epistemológico
  │
  └─ L7: Síntese final
      └─ Bloco [AUDIT L7] com metadados auditáveis

Output: Resposta estruturada + Auditoria
```

## Licença e Citações

Baseado na arquitectura de pesquisa:
- **da Costa & Abe**: Lógica Paraconsistente Anotada
- **Russell**: Teoria da Correspondência
- **Kant**: Tábua dos Juízos
- **Aristóteles**: Categorias e conceitos

## Contato e Contribuições

Para sugestões, melhorias ou reportar bugs, consulte a documentação completa do projeto Doninha IA.

---

**Versão**: 1.0  
**Última atualização**: Junho 2026  
**Status**: Pronto para produção

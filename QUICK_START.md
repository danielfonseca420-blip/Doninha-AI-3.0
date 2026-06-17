# 🎯 DONINHA MIDDLEWARE — QUICK START

## O que foi criado?

Um **arquivo Python único e self-contained** (`doninha_middleware.py`) que consolida todo o middleware Doninha com as 7 camadas epistemológicas.

## 📁 Arquivos no Desktop

- **doninha_middleware.py** (40 KB) — Middleware completo
- **DONINHA_MIDDLEWARE_README.md** — Documentação completa

## ⚡ Começar em 2 minutos

### 1. Instalação básica
```bash
# Sem APIs (modo fallback — recomendado para começar)
pip install requests
```

### 2. Primeiro código
```python
from doninha_middleware import DoninhaMiddleware

# Criar middleware
middleware = DoninhaMiddleware()

# Processar pergunta
response = middleware.process("O que é verdade?")

# Ver resposta
print(response)
```

### 3. Com LLM (OpenAI)
```python
from doninha_middleware import DoninhaMiddleware

middleware = DoninhaMiddleware({
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "sua-chave-aqui"
})

response = middleware.process("Qual é a diferença entre conhecimento e crença?")
print(response)
```

## 🔑 APIs Suportadas

| Provider | Instalação | Config |
|----------|-----------|--------|
| OpenAI | `pip install openai` | `provider: "openai"` |
| Anthropic | `pip install anthropic` | `provider: "anthropic"` |
| Google | `pip install google-generativeai` | `provider: "gemini"` |
| Ollama | `pip install ollama` | `provider: "ollama"` |
| Nenhuma | Não precisa | `provider: "fallback"` |

## 📚 Estrutura (7 Camadas)

```
Pergunta
  ↓
L1: Extrai conceitos semânticos
  ↓
L2: Gera juízos kantianos
  ↓
L3: Avalia paraconsistentemente (μ/λ)
  ↓
L4: Sintetiza via correspondência russelliana
  ↓
L5: Gera resposta com LLM
  ↓
L6: Refina o texto
  ↓
L7: Síntese final com auditoria
  ↓
Resposta estruturada + [AUDIT L7]
```

## 💬 Exemplos Rápidos

### Pergunta simples
```python
middleware = DoninhaMiddleware()
response = middleware.process("O que é lógica?")
```

### Chat
```python
messages = [
    {"role": "user", "content": "Olá"},
    {"role": "assistant", "content": "Olá!"},
    {"role": "user", "content": "Como você funciona?"}
]
response = middleware.chat(messages)
```

### Mudar LLM em tempo real
```python
middleware.set_config({
    "provider": "anthropic",
    "model": "claude-3-opus",
    "api_key": "sk-ant-..."
})

response = middleware.process("Pergunta")
```

### Ver status
```python
print(middleware.get_status())
# {'provider': 'openai', 'model': 'gpt-4', 'status': 'operacional', ...}
```

## 🎓 O que você ganha

✅ **7 camadas epistemológicas** — Raciocínio estruturado Aristotélico + Kantiano + Russell  
✅ **Lógica paraconsistente** — 12 estados lógicos com μ/λ  
✅ **Multi-LLM** — OpenAI, Claude, Gemini, Ollama com fallback automático  
✅ **Auditoria** — Blocos [AUDIT L7] para rastreabilidade completa  
✅ **Self-contained** — Um único arquivo, 40 KB, sem dependências externas  
✅ **Pronto para produção** — Type hints, logging, tratamento de erros robusto  

## 📖 Documentação Completa

Veja `DONINHA_MIDDLEWARE_README.md` para:
- Instalação detalhada
- 6+ exemplos de uso
- API completa
- Estados lógicos (12)
- Troubleshooting
- Arquitetura visual
- Casos de uso reais

## 🚀 Próximos passos

1. Coloque `doninha_middleware.py` em seu projeto
2. `from doninha_middleware import DoninhaMiddleware`
3. Configure sua API key (OpenAI, Anthropic, etc.)
4. Comece a usar conforme exemplos acima
5. Leia `DONINHA_MIDDLEWARE_README.md` para casos avançados

## ❓ Suporte

- **Erro de importação?** Verifique se o arquivo está no PATH
- **API key inválida?** Confirme suas credenciais
- **Ollama não funciona?** Execute `ollama serve` em outro terminal
- **Dúvidas?** Consulte DONINHA_MIDDLEWARE_README.md

---

**Versão**: 1.0  
**Status**: ✅ Pronto para produção  
**Data**: Junho 2026

Bom uso! 🎉

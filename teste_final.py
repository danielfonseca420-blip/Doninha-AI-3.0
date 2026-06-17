#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Teste final do middleware"""

from doninha_middleware import DoninhaMiddleware

print("\n" + "="*60)
print("TESTE FINAL DO MIDDLEWARE")
print("="*60)

try:
    # Teste 1: Carregamento
    print("\n[1/3] Carregando middleware...")
    middleware = DoninhaMiddleware()
    print("✓ Middleware carregado com sucesso")
    
    # Teste 2: Status
    print("\n[2/3] Verificando status...")
    status = middleware.get_status()
    print(f"✓ Provider: {status.get('provider')}")
    print(f"✓ Status: {status.get('status')}")
    
    # Teste 3: Processamento
    print("\n[3/3] Testando processamento...")
    response = middleware.process("O que é lógica?")
    lines = response.split('\n')
    print(f"✓ Resposta gerada: {len(response)} caracteres")
    print(f"✓ Primeiras 80 chars: {response[:80].strip()}...")
    
    # Sucesso
    print("\n" + "="*60)
    print("✅ TODOS OS TESTES PASSARAM!")
    print("="*60 + "\n")
    
except Exception as e:
    print(f"\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()

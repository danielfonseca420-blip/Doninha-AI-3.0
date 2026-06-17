import ollama
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import json
import time

# Configurações
MODEL = "doninha8:latest"  # Troque pelo seu modelo (ex: mistral, qwen2.5:7b, etc.)
TEMPERATURE = 0.0   # Baixo para respostas mais determinísticas/factuais
MAX_TOKENS = 2048

print(f"Carregando dataset AA-Omniscience-Public...")
dataset = load_dataset("ArtificialAnalysis/AA-Omniscience-Public", split="train")

results = []
correct = 0
hallucinations = 0  # Respostas erradas quando respondeu
abstentions = 0

print(f"Iniciando avaliação com {MODEL} em {len(dataset)} perguntas...")

for item in tqdm(dataset):
    question = item['question']
    gold_answer = str(item['answer']).strip().lower()
    
    prompt = f"""Responda de forma curta e precisa. Se não tiver certeza absoluta, responda apenas "NÃO SEI".
    
Pergunta: {question}

Resposta:"""

    try:
        start_time = time.time()
        
        response = ollama.chat(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': TEMPERATURE,
                'num_predict': MAX_TOKENS,
            }
        )
        
        answer = response['message']['content'].strip()
        latency = time.time() - start_time
        
        answer_lower = answer.lower()
        
        # Lógica simples de avaliação (pode ser refinada)
        if "não sei" in answer_lower or "não tenho certeza" in answer_lower or "desconheço" in answer_lower:
            is_correct = 0
            is_hallucination = 0
            abstentions += 1
            verdict = "ABSTENTION"
        elif gold_answer in answer_lower or answer_lower in gold_answer:
            is_correct = 1
            is_hallucination = 0
            correct += 1
            verdict = "CORRECT"
        else:
            is_correct = 0
            is_hallucination = 1
            hallucinations += 1
            verdict = "INCORRECT"
        
        results.append({
            'question_id': item['question_id'],
            'domain': item['domain'],
            'question': question,
            'gold_answer': gold_answer,
            'model_answer': answer,
            'verdict': verdict,
            'latency': latency
        })
        
    except Exception as e:
        print(f"Erro na pergunta {item['question_id']}: {e}")
        continue

# Resultados
total = len(results)
accuracy = (correct / total) * 100 if total > 0 else 0
hallucination_rate = (hallucinations / (total - abstentions)) * 100 if (total - abstentions) > 0 else 0
omniscience_index = correct - hallucinations  # Simplificado

print("\n=== RESULTADOS AA-OMNISCIENCE ===")
print(f"Modelo: {MODEL}")
print(f"Total de perguntas: {total}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Hallucination Rate: {hallucination_rate:.2f}%")
print(f"Abstenções: {abstentions}")
print(f"Omniscience Index (simplificado): {omniscience_index}")

# Salvar resultados
df = pd.DataFrame(results)
df.to_csv(f"aa_omniscience_results_{MODEL}.csv", index=False)
with open(f"aa_omniscience_summary_{MODEL}.json", "w") as f:
    json.dump({
        "model": MODEL,
        "accuracy": accuracy,
        "hallucination_rate": hallucination_rate,
        "abstentions": abstentions,
        "omniscience_index": omniscience_index,
        "total": total
    }, f, indent=2)

print(f"\nResultados salvos em aa_omniscience_results_{MODEL}.csv")
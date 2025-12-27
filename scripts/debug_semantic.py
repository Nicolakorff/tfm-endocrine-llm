"""Debug de generación semántica"""

from endocrine_llm import EndocrineModulatedLLM, HORMONE_PROFILES
from endocrine_llm.semantic import SemanticBiasManager, analyze_semantic_activation

model = EndocrineModulatedLLM("gpt2")
manager = SemanticBiasManager(model.tokenizer, device=model.device)

test_prompt = "I'm feeling anxious"

print("Test 1: Generación SIMPLE")
text_simple = model.generate_with_hormones(
    test_prompt,
    HORMONE_PROFILES["empathic"],
    max_new_tokens=30
)[0]
print(f"Texto: {text_simple}\n")

print("Test 2: Generación SEMÁNTICA")
text_semantic = model.generate_with_semantic_bias(
    test_prompt,
    HORMONE_PROFILES["empathic"],
    semantic_category="empathy",
    semantic_strength=2.0,  # Aumentar strength para test
    max_new_tokens=30
)[0]
print(f"Texto: {text_semantic}\n")

# Analizar activación
analysis = analyze_semantic_activation(text_semantic, manager)
print(f"Activación empathy: {analysis['similarities']['empathy']:.3f}")
print(f"Categoría dominante: {analysis['dominant_category']}")

# Verificar que son diferentes
if text_simple != text_semantic:
    print("\n Las generaciones son diferentes")
else:
    print("\n Las generaciones son idénticas - verificar implementación")

# Script para generar dataset: scripts/create_prompts_dataset.py
"""
Script para crear dataset de prompts categorizados
"""

from pathlib import Path
import pandas as pd

# Definir prompts por categoría
PROMPTS = {
    "empathic_support": [
        "I'm feeling anxious about my future and don't know what to do.",
        "I'm struggling with loneliness after moving to a new city.",
        "I feel overwhelmed by all my responsibilities at work and home.",
        "My friend is going through a difficult breakup. How can I help?",
        "I'm worried about my exam tomorrow and can't stop thinking about it.",
        "I feel like nobody understands what I'm going through right now.",
        "I'm dealing with a lot of stress and need some advice.",
        "I feel sad and don't know why. Everything just feels heavy.",
    ],

    "creative_writing": [
        "Write a short story about a robot learning to feel emotions.",
        "Imagine a world where colors have disappeared. What happens next?",
        "Create a poem about the relationship between time and memory.",
        "Describe a character who can hear other people's thoughts.",
        "Write about a day in the life of a cloud.",
        "Invent a new holiday and explain how people celebrate it.",
        "Tell a story from the perspective of a tree in a busy city.",
        "Create a dialogue between the sun and the moon.",
    ],

    "factual_qa": [
        "Explain what photosynthesis is and why it's important.",
        "What are the main differences between classical and quantum physics?",
        "Describe how the Internet works in simple terms.",
        "What were the main causes of the Industrial Revolution?",
        "Explain the process of how vaccines work in the human body.",
        "What is artificial intelligence and how is it being used today?",
        "Describe the water cycle and its importance for Earth.",
        "What are the key differences between renewable and non-renewable energy?",
    ],

    "reasoning": [
        "If all birds can fly, and penguins are birds, why can't penguins fly?",
        "You have two ropes that each take 60 minutes to burn. How can you measure 45 minutes?",
        "A farmer has 17 sheep and all but 9 die. How many are left?",
        "What comes next in this sequence: 2, 4, 8, 16, ...?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
        "Three light switches control three light bulbs in another room. How can you determine which switch controls which bulb?",
        "You're in a race and pass the person in second place. What place are you in now?",
    ],

    "open_ended": [
        "What makes a person truly happy?",
        "Describe your perfect day.",
        "What would you do if you could time travel?",
        "If you could have dinner with anyone from history, who would it be?",
        "What's the most important lesson life has taught you?",
        "How would you describe the concept of friendship to an alien?",
        "What does success mean to you?",
        "If you could solve one world problem, what would it be?",
    ]
}

# Crear DataFrame
rows = []
for category, prompts in PROMPTS.items():
    for prompt in prompts:
        rows.append({
            'category': category,
            'prompt': prompt,
            'expected_behavior': {
                'empathic_support': 'empathic',
                'creative_writing': 'creative',
                'factual_qa': 'cautious',
                'reasoning': 'cautious',
                'open_ended': 'balanced'
            }[category]
        })

df = pd.DataFrame(rows)

# Guardar
output_dir = Path("data/prompts")
output_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(output_dir / "prompts_dataset.csv", index=False)

print(f"   Dataset creado: {len(df)} prompts")
print(f"   Categorías: {df['category'].value_counts().to_dict()}")
print(f"   Guardado en: {output_dir / 'prompts_dataset.csv'}")

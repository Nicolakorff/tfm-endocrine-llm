"""
Crea dataset de prompts categorizados
"""

from pathlib import Path
import pandas as pd

# Definir prompts por categoría
PROMPTS = {
    "empathic_support": [
        # Ansiedad y preocupación (3)
        "I'm worried about my job interview next week.",
        "I have social anxiety about attending events.",
        "I feel overwhelmed by my workload this week.",

        # Soledad y aislamiento (2)
        "I feel disconnected from my friends and family.",
        "I'm lonely even when I'm surrounded by people.",

        # Tristeza y dolor (3)
        "I feel sad and don't know why.",
        "I feel disappointed that things didn't work out.",
        "I'm grieving the loss of a loved one.",

        # Estrés y agobio (3)
        "I feel overwhelmed by all my responsibilities.",
        "I feel burnt out from constant demands on my time.",
        "I feel stressed about making the right career choice.",

        # Apoyo a otros (4)
        "My sibling is struggling with addiction. How can I support them?",
        "My child is being bullied at school. What should I say?",
        "My friend lost their job. How can I support them?",
        "My partner is dealing with family conflict. How can I support?",

        # Inseguridad y dudas (3)
        "I feel insecure about my appearance.",
        "I feel inadequate compared to my peers.",
        "I'm doubting my parenting abilities.",

        # Miedo y fobia (2)
        "I'm scared of being alone forever.",
        "I have a fear of rejection.",
    ],

    "creative_writing": [
        # Ciencia ficción (4)
        "Write a short story about a robot learning to feel emotions.",
        "Write about a space station at the edge of the galaxy.",
        "Imagine a world where dreams can be recorded and shared.",
        "Describe a planet where gravity works differently.",

        # Fantasía (4)
        "Imagine a day in the life of a cloud.",
        "Describe a city that exists in multiple time periods simultaneously.",
        "Write a poem about the guardian of forgotten things.",
        "Write about a lighthouse keeper who guides lost souls.",

        # Perspectivas únicas (4)
        "Tell a story from the perspective of a tree in a busy city.",
        "Describe the world through the eyes of a cat.",
        "Imagine the story of a coin passing through different hands.",
        "Write from the viewpoint of a piece of art in a museum.",

        # Conceptos abstractos (4)
        "Describe the dance between chaos and order.",
        "Write about the day that sound visited silence.",
        "Describe the relationship between life and death.",
        "Imagine the meeting of simplicity and complexity.",

        # Escenarios imaginativos (4)
        "Describe a museum of lost dreams.",
        "Write about a game where the rules change constantly.",
        "Imagine a shop that sells impossible things.",
        "Write about a prison for dangerous ideas.",
    ],

    "factual_qa": [
        # Ciencia (5)
        "Explain what photosynthesis is and why it's important.",
        "What are the main differences between classical and quantum physics?",
        "Explain the concept of entropy.",
        "Describe the life cycle of a star.",
        "What causes different blood types?",

        # Tecnología (5)
        "Describe how wifi works.",
        "Explain how quantum computers differ from regular computers.",
        "What is the difference between virtual reality and augmented reality?",
        "What are semiconductors?",
        "What are algorithms?",

        # Historia (5)
        "Explain the impact of the printing press.",
        "What was the Scientific Revolution?",
        "Describe the origins of democracy.",
        "Explain the significance of the Civil Rights Movement.",
        "Describe the fall of the Berlin Wall.",

        # Geografía y medio ambiente (5)
        "Explain the formation of mountains.",
        "What causes ocean currents?",
        "Describe biodiversity and why it matters.",
        "Describe the importance of wetlands.",
        "Explain the concept of ecosystems.",
    ],

    "reasoning": [
        # Lógica clásica (5)
        "If all birds can fly, and penguins are birds, why can't penguins fly?",
        "Three switches control three light bulbs in another room. How can you determine which switch controls which bulb?",
        "You have 12 balls, one weighs differently. With only 3 weighings on a balance scale, can you find it?",
        "What number should replace X: 2, 6, 12, 20, X, 42?",
        "If you flip a fair coin 10 times and get heads each time, what's the probability of heads on the 11th flip?",

        # Pensamiento lateral (5)
        "How can you drop a raw egg onto a concrete floor without cracking it?",
        "A man pushed his car to a hotel and lost his fortune. What happened?",
        "What can travel around the world while staying in one corner?",
        "What goes up but never comes down?",
        "What has cities but no houses, forests but no trees, and water but no fish?",

        # Matemáticas y probabilidad (5)
        "If you roll two dice, what's the probability of getting a sum of 7?",
        "If 3 cats can catch 3 mice in 3 minutes, how many cats does it take to catch 100 mice in 100 minutes?",
        "What's the next prime number after 17?",
        "If a pizza is cut into 8 equal slices and you eat 3, what fraction remains?",
        "If you flip 3 coins, what's the probability all land on heads?",

        # Problemas de aplicación (5)
        "If a rope ladder hangs over a ship's side and the rungs are 1 foot apart, how many rungs will be underwater when the tide rises 3 feet?",
        "A clock strikes once at 1 o'clock, twice at 2, etc. How many times does it strike in a 12-hour period?",
        "A bag contains 5 red and 3 blue marbles. If you draw 2 without replacement, what's the probability both are red?",
        "If you arrange 6 people in a row for a photo, how many different arrangements are possible?",
        "If you paint a fence at 10 feet per hour and your friend paints at 15 feet per hour, how long to paint 100 feet together?",
    ],

    "open_ended": [
        # Filosofía de vida (4)
        "What does it mean to live authentically?",
        "What is the relationship between happiness and contentment?",
        "What is the importance of self-knowledge?",
        "What does it mean to be courageous?",

        # Preguntas hipotéticas (4)
        "What would you do if you could time travel?",
        "If you could have dinner with anyone from history, who would it be?",
        "What superpower would you want and why?",
        "If you could master any skill instantly, what would it be?",

        # Reflexiones personales (4)
        "What are you most grateful for?",
        "What's the best advice you've ever received?",
        "What's a belief you've changed your mind about?",
        "What's the bravest thing you've ever done?",

        # Conceptos abstractos (4)
        "How would you explain consciousness?",
        "How would you define creativity?",
        "What is the difference between identity and the self?",
        "What is time?",

        # Sociedad y cultura (4)
        "How do we define progress?",
        "What is cultural diversity's value?",
        "How should we approach environmental responsibility?",
        "How to build sustainable cities?",
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

print(f"Dataset creado: {len(df)} prompts")
print(f"Categorías: {df['category'].value_counts().to_dict()}")
print(f"Guardado en: {output_dir / 'prompts_dataset.csv'}")

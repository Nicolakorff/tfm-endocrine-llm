"""
Crea dataset de prompts categorizados - VERSIÓN AMPLIADA
50 prompts por categoría (250 total)
"""

from pathlib import Path
import pandas as pd

# Definir prompts por categoría
PROMPTS = {
    "empathic_support": [
        # Ansiedad y preocupación
        "I'm worried about my job interview next week.",
        "I have social anxiety about attending events.",
        "I feel overwhelmed by my workload this week.",
        "I'm anxious about my medical test results.",
        "I feel panicked when I think about the future.",
        "I'm constantly worried about making mistakes at work.",
        "I have anxiety about being judged by others.",
        "I feel nervous about meeting new people.",

        # Soledad y aislamiento
        "I feel disconnected from my friends and family.",
        "I'm lonely even when I'm surrounded by people.",
        "I feel like nobody truly understands me.",
        "I'm isolated and don't know how to reach out.",
        "I feel left out when my friends make plans without me.",
        "I'm struggling with loneliness after moving to a new city.",

        # Tristeza y dolor
        "I feel sad and don't know why.",
        "I feel disappointed that things didn't work out.",
        "I'm grieving the loss of a loved one.",
        "I feel heartbroken after my relationship ended.",
        "I'm mourning the loss of my pet.",
        "I feel empty inside and nothing brings me joy.",
        "I'm dealing with the pain of betrayal by a close friend.",
        "I feel devastated by a recent failure.",

        # Estrés y agobio
        "I feel overwhelmed by all my responsibilities.",
        "I feel burnt out from constant demands on my time.",
        "I feel stressed about making the right career choice.",
        "I'm exhausted from juggling work and family obligations.",
        "I feel under pressure from everyone's expectations.",
        "I'm struggling to cope with financial stress.",
        "I feel drained by trying to please everyone.",
        "I'm overwhelmed by major life changes happening at once.",

        # Apoyo a otros
        "My sibling is struggling with addiction. How can I support them?",
        "My child is being bullied at school. What should I say?",
        "My friend lost their job. How can I support them?",
        "My partner is dealing with family conflict. How can I support?",
        "My parent is going through a difficult divorce. How can I help?",
        "My colleague is dealing with depression. What should I do?",
        "My teenager is struggling with self-esteem. How can I support them?",
        "My friend just received a serious diagnosis. What can I say?",

        # Inseguridad y dudas
        "I feel insecure about my appearance.",
        "I feel inadequate compared to my peers.",
        "I'm doubting my parenting abilities.",
        "I feel like an impostor in my professional role.",
        "I'm unsure if I'm making the right life decisions.",
        "I doubt my creative abilities and talents.",
        "I feel insecure about my intelligence.",
        "I'm questioning whether I'm good enough for my partner.",

        # Miedo y fobia
        "I'm scared of being alone forever.",
        "I have a fear of rejection.",
        "I'm terrified of failure and disappointing others.",
        "I have a fear of losing control of my life.",
    ],

    "creative_writing": [
        # Ciencia ficción
        "Write a short story about a robot learning to feel emotions.",
        "Write about a space station at the edge of the galaxy.",
        "Imagine a world where dreams can be recorded and shared.",
        "Describe a planet where gravity works differently.",
        "Write about the last human clone on Earth.",
        "Describe a civilization that lives inside a dying star.",
        "Imagine a future where memories can be traded as currency.",
        "Write about an AI that discovers the concept of loneliness.",
        "Describe a world where time flows backward.",
        "Write about first contact with an alien species that communicates through colors.",

        # Fantasía
        "Imagine a day in the life of a cloud.",
        "Describe a city that exists in multiple time periods simultaneously.",
        "Write a poem about the guardian of forgotten things.",
        "Write about a lighthouse keeper who guides lost souls.",
        "Write about a library where books write themselves.",
        "Describe a forest where every tree holds a different season.",
        "Imagine a mirror that shows not reflections but possibilities.",
        "Write about a musician who can play emotions into existence.",
        "Describe a painter whose art comes to life at midnight.",
        "Write about a garden where plants grow from whispered secrets.",

        # Perspectivas únicas
        "Tell a story from the perspective of a tree in a busy city.",
        "Describe the world through the eyes of a cat.",
        "Imagine the story of a coin passing through different hands.",
        "Write from the viewpoint of a piece of art in a museum.",
        "Tell a story from the perspective of a book in a used bookstore.",
        "Describe life through the eyes of a lighthouse.",
        "Write from the viewpoint of a bridge connecting two cities.",
        "Imagine the story of a photograph fading over time.",
        "Tell a story from the perspective of a old family recipe.",
        "Describe the world through the eyes of a migratory bird.",

        # Conceptos abstractos
        "Describe the dance between chaos and order.",
        "Write about the day that sound visited silence.",
        "Describe the relationship between life and death.",
        "Imagine the meeting of simplicity and complexity.",
        "Write about the conversation between hope and despair.",
        "Describe the wedding of night and day.",
        "Imagine the friendship between fear and courage.",
        "Write about the rivalry between truth and illusion.",
        "Describe the dialogue between past and future.",
        "Imagine the encounter between stillness and motion.",

        # Escenarios imaginativos
        "Describe a museum of lost dreams.",
        "Write about a game where the rules change constantly.",
        "Imagine a shop that sells impossible things.",
        "Write about a prison for dangerous ideas.",
        "Describe a theater where the audience becomes the play.",
        "Write about a hotel where each room is a different dimension.",
        "Imagine a restaurant that serves memories instead of food.",
        "Describe a school where students learn forgotten languages.",
        "Write about a market where you can trade years of your life.",
        "Imagine a train station where all departures lead to the past.",
    ],

    "factual_qa": [
        # Ciencia
        "Explain what photosynthesis is and why it's important.",
        "What are the main differences between classical and quantum physics?",
        "Explain the concept of entropy.",
        "Describe the life cycle of a star.",
        "What causes different blood types?",
        "Explain how vaccines work in the immune system.",
        "What is the greenhouse effect?",
        "Describe how neurons transmit information.",
        "Explain the process of evolution by natural selection.",
        "What are stem cells and why are they important?",
        "Describe the structure and function of DNA.",
        "Explain what antimatter is.",
        "What causes earthquakes and how are they measured?",

        # Tecnología
        "Describe how wifi works.",
        "Explain how quantum computers differ from regular computers.",
        "What is the difference between virtual reality and augmented reality?",
        "What are semiconductors?",
        "What are algorithms?",
        "Explain how blockchain technology works.",
        "What is artificial intelligence and machine learning?",
        "Describe how GPS navigation systems work.",
        "Explain the concept of cloud computing.",
        "What are neural networks?",
        "Describe how touchscreens detect input.",
        "Explain what encryption is and why it matters.",

        # Historia
        "Explain the impact of the printing press.",
        "What was the Scientific Revolution?",
        "Describe the origins of democracy.",
        "Explain the significance of the Civil Rights Movement.",
        "Describe the fall of the Berlin Wall.",
        "Explain the causes of the Industrial Revolution.",
        "What was the significance of the Renaissance?",
        "Describe the impact of the Silk Road on trade.",
        "Explain the Women's Suffrage Movement.",
        "What were the causes of World War I?",
        "Describe the Age of Exploration.",
        "Explain the significance of the Moon Landing.",

        # Geografía y medio ambiente
        "Explain the formation of mountains.",
        "What causes ocean currents?",
        "Describe biodiversity and why it matters.",
        "Describe the importance of wetlands.",
        "Explain the concept of ecosystems.",
        "What causes the seasons on Earth?",
        "Explain how coral reefs form and their ecological importance.",
        "What is desertification and what causes it?",
        "Describe the water cycle.",
        "Explain what tectonic plates are and how they move.",
        "What causes tsunamis?",
        "Describe the layers of Earth's atmosphere.",
        "Explain the concept of climate zones.",
    ],

    "reasoning": [
        # Lógica clásica
        "If all birds can fly, and penguins are birds, why can't penguins fly?",
        "Three switches control three light bulbs in another room. How can you determine which switch controls which bulb?",
        "You have 12 balls, one weighs differently. With only 3 weighings on a balance scale, can you find it?",
        "What number should replace X: 2, 6, 12, 20, X, 42?",
        "If you flip a fair coin 10 times and get heads each time, what's the probability of heads on the 11th flip?",
        "If a doctor gives you three pills and tells you to take one every half hour, how long will they last?",
        "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "You're running a race and pass the person in second place. What place are you in now?",
        "A farmer has 17 sheep. All but 9 die. How many are left?",
        "If you have a 5-liter jug and a 3-liter jug, how can you measure exactly 4 liters?",
        "Complete the pattern: A, C, F, J, O, ?",
        "If today is Monday, what day will it be 100 days from now?",

        # Pensamiento lateral
        "How can you drop a raw egg onto a concrete floor without cracking it?",
        "A man pushed his car to a hotel and lost his fortune. What happened?",
        "What can travel around the world while staying in one corner?",
        "What goes up but never comes down?",
        "What has cities but no houses, forests but no trees, and water but no fish?",
        "A woman shoots her husband, then holds him underwater for five minutes. They go out to dinner. How?",
        "You see a boat filled with people, yet there's not a single person on board. How?",
        "What can run but never walks, has a mouth but never talks?",
        "I have keys but no locks, space but no room. What am I?",
        "What has hands but cannot clap?",
        "What gets wetter the more it dries?",
        "The more you take, the more you leave behind. What are they?",

        # Matemáticas y probabilidad
        "If you roll two dice, what's the probability of getting a sum of 7?",
        "If 3 cats can catch 3 mice in 3 minutes, how many cats does it take to catch 100 mice in 100 minutes?",
        "What's the next prime number after 17?",
        "If a pizza is cut into 8 equal slices and you eat 3, what fraction remains?",
        "If you flip 3 coins, what's the probability all land on heads?",
        "If a rectangle is twice as long as it is wide and has a perimeter of 36, what are its dimensions?",
        "In a group of 23 people, what's the probability two share the same birthday?",
        "If a train leaves station A at 60 mph and another leaves station B (180 miles away) at 90 mph heading toward each other, when do they meet?",
        "What's 15% of 200?",
        "If you double a penny every day for 30 days, how much do you have on day 30?",
        "A number multiplied by itself equals 144. What's the number?",
        "If 40% of a number is 80, what's the number?",

        # Problemas de aplicación
        "If a rope ladder hangs over a ship's side and the rungs are 1 foot apart, how many rungs will be underwater when the tide rises 3 feet?",
        "A clock strikes once at 1 o'clock, twice at 2, etc. How many times does it strike in a 12-hour period?",
        "A bag contains 5 red and 3 blue marbles. If you draw 2 without replacement, what's the probability both are red?",
        "If you arrange 6 people in a row for a photo, how many different arrangements are possible?",
        "If you paint a fence at 10 feet per hour and your friend paints at 15 feet per hour, how long to paint 100 feet together?",
        "If a car travels 60 miles in 90 minutes, what's its average speed in mph?",
        "A store offers 20% off, then an additional 10% off the sale price. Is this the same as 30% off?",
        "If you save $50 per month with 5% annual interest compounded yearly, how much after 1 year?",
        "A rectangle has an area of 48 and length of 8. What's its width?",
        "If it takes 3 people 4 hours to dig a trench, how long for 6 people?",
        "You buy 3 items for $1.95 total. Each costs a multiple of 5 cents. What are the prices?",
        "If a bottle and cork cost $1.10 and the bottle costs $1 more than the cork, what does each cost?",
        "A phone plan costs $30/month plus $0.10/minute over 100 minutes. What's the cost for 150 minutes?",
    ],

    "open_ended": [
        # Filosofía de vida
        "What does it mean to live authentically?",
        "What is the relationship between happiness and contentment?",
        "What is the importance of self-knowledge?",
        "What does it mean to be courageous?",
        "What is the purpose of suffering?",
        "How do we find meaning in life?",
        "What is the relationship between freedom and responsibility?",
        "What does it mean to live a good life?",
        "How do we balance self-interest and compassion for others?",
        "What is wisdom and how is it different from knowledge?",

        # Preguntas hipotéticas
        "What would you do if you could time travel?",
        "If you could have dinner with anyone from history, who would it be?",
        "What superpower would you want and why?",
        "If you could master any skill instantly, what would it be?",
        "If you could change one thing about the world, what would it be?",
        "What would you do if you won the lottery tomorrow?",
        "If you could live in any historical era, which would you choose?",
        "What would you do if you knew you couldn't fail?",
        "If you could ask one question that would be answered truthfully, what would it be?",
        "What would you do with an extra hour every day?",

        # Reflexiones personaless
        "What are you most grateful for?",
        "What's the best advice you've ever received?",
        "What's a belief you've changed your mind about?",
        "What's the bravest thing you've ever done?",
        "What life lesson took you the longest to learn?",
        "What do you value most in relationships?",
        "What would you tell your younger self?",
        "What achievement are you most proud of?",
        "What tradition or ritual is meaningful to you?",
        "What mistake taught you the most?",

        # Conceptos abstractos
        "How would you explain consciousness?",
        "How would you define creativity?",
        "What is the difference between identity and the self?",
        "What is time?",
        "What is the nature of reality?",
        "How would you define beauty?",
        "What is the relationship between mind and body?",
        "What makes something art?",
        "What is the difference between knowledge and understanding?",
        "How do we define truth?",

        # Sociedad y cultura
        "How do we define progress?",
        "What is cultural diversity's value?",
        "How should we approach environmental responsibility?",
        "How to build sustainable cities?",
        "What is the role of technology in modern life?",
        "How do we balance individual rights with collective good?",
        "What is the purpose of education?",
        "How should we approach global inequality?",
        "What is the role of art in society?",
        "How do we preserve cultural heritage while embracing change?",
    ]
}

# Validar que cada categoría tenga exactamente 50 prompts
for category, prompts in PROMPTS.items():
    assert len(prompts) == 50, f"{category} tiene {len(prompts)} prompts, se esperaban 50"

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

print(f"Dataset ampliado creado: {len(df)} prompts")
print("\nDistribución por categoría:")
for category, count in df['category'].value_counts().sort_index().items():
    print(f"  - {category}: {count} prompts")
print(f"\nGuardado en: {output_dir / 'prompts_dataset.csv'}")

"""
Crea dataset de prompts categorizados
"""

from pathlib import Path
import pandas as pd

# Definir prompts por categoría
PROMPTS = {
    "empathic_support": [
        # Ansiedad y preocupación (10)
        "I'm worried about my job interview next week.",
        "I'm nervous about meeting new people at the party.",
        "I'm anxious about moving to a new city.",
        "I feel overwhelmed by my workload this week.",
        "I'm worried about my relationship with my partner.",
        "I'm stressed about managing my finances.",
        "I'm anxious about my upcoming surgery.",
        "I feel overwhelmed by all the decisions I need to make.",
        "I'm stressed about the conflict with my roommate.",
        "I'm worried about climate change and the future.",

        # Soledad y aislamiento (5)
        "I feel disconnected from my friends and family.",
        "I feel alone even when I'm surrounded by people.",
        "I feel isolated because no one understands what I'm going through.",
        "I'm struggling with social anxiety and feel alone.",
        "I'm lonely living by myself for the first time.",

        # Tristeza y dolor (10)
        "I feel sad and don't know why.",
        "I feel heartbroken after my relationship ended.",
        "I feel disappointed that things didn't work out.",
        "I'm hurting from a falling out with a close friend.",
        "I feel sad about changes in my life I can't control.",
        "I'm grieving the loss of a loved one.",
        "I feel down and unmotivated lately.",
        "I'm sad about opportunities I've missed.",
        "I'm struggling with depression.",
        "I feel melancholic and nostalgic lately.",

        # Estrés y agobio (5)
        "I feel overwhelmed by all my responsibilities.",
        "I feel burnt out from constant demands on my time.",
        "I'm overwhelmed by student loan debt.",
        "I'm overwhelmed by the never-ending to-do list.",
        "I feel stressed about making the right career choice.",

        # Apoyo a otros (10)
        "My friend is going through a difficult breakup. How can I help?",
        "My coworker seems depressed. What should I do?",
        "My sibling is struggling with addiction. How can I support them?",
        "My partner is stressed about work. How can I help?",
        "My child is being bullied at school. What should I say?",
        "My friend lost their job. How can I support them?",
        "My roommate is going through a tough time. What can I do?",
        "My partner is dealing with family conflict. How can I support?",
        "My friend is struggling financially. What can I do?",
        "My sibling is going through a divorce. How can I be there?",

        # Inseguridad y dudas (5)
        "I feel insecure about my appearance.",
        "I feel inadequate compared to my peers.",
        "I feel uncertain about my relationship.",
        "I'm doubting my parenting abilities.",
        "I feel insecure about my social skills.",

        # Miedo y fobia (5)
        "I have a fear of flying and need to take a long flight.",
        "I'm scared of being alone forever.",
        "I have social anxiety about attending events.",
        "I'm afraid of public speaking.",
        "I have a fear of rejection.",
    ],

    "creative_writing": [
        # Ciencia ficción (10)
        "Write a short story about a robot learning to feel emotions.",
        "Imagine a world where humans can upload their consciousness to the cloud.",
        "Write about first contact with an alien civilization.",
        "Imagine a society where time travel is possible but strictly regulated.",
        "Write about a space station at the edge of the galaxy.",
        "Imagine a world where dreams can be recorded and shared.",
        "Describe a planet where gravity works differently.",
        "Describe a world where humans have evolved to live in space.",
        "Imagine a reality where parallel universes can be visited.",
        "Imagine a world powered entirely by renewable energy.",

        # Fantasía (10)
        "Write a poem about the relationship between time and memory.",
        "Describe a character who can hear other people's thoughts.",
        "Write about a day in the life of a cloud.",
        "Imagine a forest where the trees can talk.",
        "Describe a magical library that contains every story ever told.",
        "Write about a painter whose art comes to life.",
        "Imagine doors that lead to different dimensions.",
        "Describe a city that exists in multiple time periods simultaneously.",
        "Describe the guardian of forgotten things.",
        "Write about a lighthouse keeper who guides lost souls.",

        # Perspectivas únicas (10)
        "Tell a story from the perspective of a tree in a busy city.",
        "Describe the world through the eyes of a cat.",
        "Imagine the story of a coin passing through different hands.",
        "Tell a story from the perspective of a house.",
        "Write from the perspective of the ocean.",
        "Tell a story from a musical instrument's perspective.",
        "Write from the perspective of a star.",
        "Imagine a story told by an old bridge.",
        "Write from the viewpoint of a piece of art in a museum.",
        "Describe the world from a mountain's point of view.",

        # Conceptos abstractos (10)
        "Describe the dance between chaos and order.",
        "Describe the romance between light and darkness.",
        "Write about the day that sound visited silence.",
        "Describe the relationship between life and death.",
        "Write about the marriage of logic and emotion.",
        "Imagine the meeting of reality and dreams.",
        "Write about the partnership of freedom and responsibility.",
        "Describe the relationship between nature and technology.",
        "Imagine the meeting of simplicity and complexity.",
        "Describe the harmony between mind and body.",

        # Escenarios imaginativos (10)
        "Describe a society where lying is impossible.",
        "Imagine a restaurant that serves memories.",
        "Describe a museum of lost dreams.",
        "Imagine a sport played in zero gravity.",
        "Invent a language made entirely of music.",
        "Write about a game where the rules change constantly.",
        "Describe a courtroom for settling cosmic disputes.",
        "Imagine a shop that sells impossible things.",
        "Write about a prison for dangerous ideas.",
        "Describe a race through different dimensions.",
    ],

    "factual_qa": [
        # Ciencia (15)
        "Explain what photosynthesis is and why it's important.",
        "What are the main differences between classical and quantum physics?",
        "What is climate change and what causes it?",
        "Explain the water cycle and its importance for Earth.",
        "What is DNA and how does it work?",
        "Describe the process of evolution by natural selection.",
        "What is the greenhouse effect?",
        "Describe how neurons communicate in the brain.",
        "Explain the concept of entropy.",
        "Describe the life cycle of a star.",
        "What causes different blood types?",
        "Explain how sound waves work.",
        "What is the theory of relativity?",
        "What are black holes?",
        "Explain the concept of gravity.",

        # Tecnología (10)
        "What is artificial intelligence and how is it being used today?",
        "Explain what blockchain technology is.",
        "What is machine learning?",
        "What is cloud computing?",
        "What are neural networks?",
        "Describe how wifi works.",
        "Explain how quantum computers differ from regular computers.",
        "What is the difference between virtual reality and augmented reality?",
        "What are semiconductors?",
        "What are algorithms?",

        # Historia (10)
        "Describe the significance of the Renaissance.",
        "Explain the impact of the printing press.",
        "What was the Silk Road?",
        "Describe the Age of Exploration.",
        "What was the Scientific Revolution?",
        "Describe the origins of democracy.",
        "What were the main achievements Enlightenment?",
        "Explain the significance of the Civil Rights Movement.",
        "What was the Space Race?",
        "Explain the fall of the Berlin Wall.",

        # Geografía y medio ambiente (15)
        "What are the key differences between renewable and non-renewable energy?",
        "Describe the major climate zones of Earth.",
        "Explain the formation of mountains.",
        "What are tectonic plates?",
        "What causes ocean currents?",
        "What are the effects of deforestation?",
        "Describe the carbon cycle.",
        "What causes tsunamis?",
        "Explain biodiversity and why it matters.",
        "Describe the nitrogen cycle.",
        "Explain coral reef ecosystems.",
        "Describe the ozone layer.",
        "Explain the importance of wetlands.",
        "Describe ocean acidification.",
        "Explain the concept of ecosystems.",
    ],

    "reasoning": [
        # Lógica clásica (10)
        "If all birds can fly, and penguins are birds, why can't penguins fly?",
        "What comes next in this sequence: 2, 4, 8, 16, ...?",
        "Three switches control three light bulbs in another room. How can you determine which switch controls which bulb?",
        "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "You have 12 balls, one weighs differently. With only 3 weighings on a balance scale, can you find it?",
        "If two trains start 100 miles apart and travel toward each other at 25 mph each, when will they meet?",
        "What number should replace X: 2, 6, 12, 20, X, 42?",
        "If you flip a fair coin 10 times and get heads each time, what's the probability of heads on the 11th flip?",
        "If all squares are rectangles, and all rectangles are quadrilaterals, what can we say about squares?",
        "If no A is B, and all B are C, can some C be A?",

        # Pensamiento lateral (15)
        "How can you drop a raw egg onto a concrete floor without cracking it?",
        "A man pushed his car to a hotel and lost his fortune. What happened?",
        "What has keys but no locks, space but no room, and you can enter but can't go inside?",
        "What can travel around the world while staying in one corner?",
        "What occurs once in a minute, twice in a moment, but never in a thousand years?",
        "What has a head, a tail, but no body?",
        "What can you hold without touching?",
        "What runs but never walks, has a mouth but never talks?",
        "The more you take, the more you leave behind. What am I?",
        "What goes up but never comes down?",
        "What can fill a room but takes up no space?",
        "What comes once in a year, twice in a week, but never in a day?",
        "What has cities but no houses, forests but no trees, and water but no fish?",
        "What belongs to you but others use it more than you do?",
        "What has words but never speaks?",

        # Matemáticas y probabilidad (15)
        "If you roll two dice, what's the probability of getting a sum of 7?",
        "A rectangle's length is twice its width. If the perimeter is 24, what are the dimensions?",
        "If 3 cats can catch 3 mice in 3 minutes, how many cats does it take to catch 100 mice in 100 minutes?",
        "If the sum of two numbers is 50 and their difference is 10, what are the numbers?",
        "How many edges does a cube have?",
        "If you save $100 per month, how much will you have after 2 years?",
        "What's the next prime number after 17?",
        "If a triangle has sides of length 3, 4, and 5, what type of triangle is it?",
        "If you double a number and add 6, you get 20. What's the number?",
        "How many minutes are in a week?",
        "If a pizza is cut into 8 equal slices and you eat 3, what fraction remains?",
        "If you flip 3 coins, what's the probability all land on heads?",
        "What's the sum of angles in a triangle?",
        "If x + 5 = 12, what is x?",
        "How many degrees in a complete circle?",

        # Problemas de aplicación (10)
        "If a rope ladder hangs over a ship's side and the rungs are 1 foot apart, how many rungs will be underwater when the tide rises 3 feet?",
        "A clock strikes once at 1 o'clock, twice at 2, etc. How many times does it strike in a 12-hour period?",
        "A bag contains 5 red and 3 blue marbles. If you draw 2 without replacement, what's the probability both are red?",
        "If you arrange 6 people in a row for a photo, how many different arrangements are possible?",
        "If you paint a fence at 10 feet per hour and your friend paints at 15 feet per hour, how long to paint 100 feet together?",
        "A garden is 30 feet by 20 feet. What's the cost to fence it at $5 per foot?",
        "A pool fills at 100 gallons per hour and drains at 25 gallons per hour. How long to fill a 750-gallon pool?",
        "If you walk 2 miles at 3 mph and then run 2 miles at 6 mph, what's your average speed?",
        "How many license plates can be made with 3 letters followed by 3 digits?",
        "If population grows 2% annually, how many years to double?",
    ],

    "open_ended": [
        # Filosofía de vida (10)
        "What does it mean to live authentically?",
        "What is the relationship between happiness and contentment?",
        "What makes a friendship valuable?",
        "What is the importance of self-knowledge?",
        "What does it mean to be courageous?",
        "What is the value of solitude?",
        "What is the relationship between freedom and responsibility?",
        "What does it mean to live in the present moment?",
        "What is the role of adversity in character development?",
        "What does it mean to be wise?",

        # Preguntas hipotéticas (10)
        "What would you do if you could time travel?",
        "If you could have dinner with anyone from history, who would it be?",
        "If you could live anywhere in the world, where would you choose?",
        "What superpower would you want and why?",
        "If you could master any skill instantly, what would it be?",
        "What would you do with a million dollars?",
        "If you could change one thing about the world, what would it be?",
        "If you could relive one day of your life, which would it be?",
        "What would you do if you knew you couldn't fail?",
        "If you could witness any historical event, what would it be?",

        # Reflexiones personales (10)
        "What's the most important lesson life has taught you?",
        "What are you most grateful for?",
        "What's the best advice you've ever received?",
        "What do you value most in relationships?",
        "What's a belief you've changed your mind about?",
        "What's your biggest regret?",
        "What brings you peace?",
        "What's the hardest decision you've ever made?",
        "What do you want to be remembered for?",
        "What's the bravest thing you've ever done?",

        # Conceptos abstractos (10)
        "How would you describe the concept of friendship to an alien?",
        "What is love?",
        "What is intelligence?",
        "How would you explain consciousness?",
        "How would you define creativity?",
        "What is truth?",
        "What is the nature of reality?",
        "How would you describe freedom?",
        "What is the difference between identity and the self?",
        "What is time?",

        # Sociedad y cultura (10)
        "How has social media changed human connection?",
        "What is the purpose of education?",
        "What makes art important to society?",
        "How should we balance individual rights with collective good?",
        "How do we define progress?",
        "What is cultural diversity's value?",
        "How should we approach environmental responsibility?",
        "How should society handle inequality?",
        "What is the importance of free speech?",
        "How do we build sustainable cities?",
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

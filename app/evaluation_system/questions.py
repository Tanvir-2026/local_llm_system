"""20 predefined evaluation questions.

`keywords` is a tuple of groups. Each group is a tuple of acceptable alternatives;
an answer is fully correct when at least one alternative from EVERY group appears.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: int
    category: str
    question: str
    reference: str
    keywords: tuple[tuple[str, ...], ...]


QUESTIONS: list[Question] = [
    # --- factual ---
    Question(1, "factual", "What is the capital of Australia?", "Canberra.", (("canberra",),)),
    Question(2, "factual", "Who wrote the play 'Romeo and Juliet'?", "William Shakespeare.", (("shakespeare",),)),
    Question(3, "factual", "What is the chemical symbol for gold?", "Au.", (("au",),)),
    Question(4, "factual", "Which is the longest river in Africa?", "The Nile.", (("nile",),)),
    Question(5, "factual", "In which year did World War II end?", "1945.", (("1945",),)),
    # --- science ---
    Question(6, "science", "What gas do plants absorb from the atmosphere for photosynthesis?", "Carbon dioxide (CO2).", (("carbon dioxide", "co2", "co₂"),)),
    Question(7, "science", "What is the boiling point of water at sea level in degrees Celsius?", "100 degrees Celsius.", (("100",),)),
    # --- math ---
    Question(8, "math", "What is 15% of 240?", "36.", (("36",),)),
    Question(9, "math", "If a train travels 60 km in 45 minutes, what is its average speed in km/h?", "80 km/h.", (("80",),)),
    Question(10, "math", "What is the next number in the sequence 2, 6, 12, 20, 30?", "42 (the differences grow by 2).", (("42",),)),
    # --- reasoning ---
    Question(11, "reasoning", "Tom is taller than Sam. Sam is taller than Ali. Who is the shortest?", "Ali.", (("ali",),)),
    Question(12, "reasoning", "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?", "5 cents ($0.05).", (("5 cents", "$0.05", "0.05", "five cents", "5¢"),)),
    # --- coding / tech ---
    Question(13, "coding", "In Python, which built-in function returns the number of items in a list?", "len().", (("len",),)),
    Question(14, "coding", "What does HTTP status code 404 mean?", "Not Found: the requested resource does not exist.", (("not found",),)),
    Question(15, "coding", "What is the time complexity of binary search on a sorted array?", "O(log n).", (("o(log n)", "o(log2 n)", "o(log(n))", "log n", "log(n)", "logarithmic"),)),
    Question(16, "tech", "What does the acronym RAG stand for in the context of AI?", "Retrieval-Augmented Generation.", (("retrieval-augmented generation", "retrieval augmented generation"),)),
    Question(17, "tech", "What does JSON stand for?", "JavaScript Object Notation.", (("javascript object notation",),)),
    # --- instruction following ---
    Question(18, "instruction", "List the three primary colors of light (the additive color model).", "Red, green and blue.", (("red",), ("green",), ("blue",))),
    Question(19, "instruction", "Translate 'Good morning' into Spanish.", "Buenos días.", (("buenos días", "buenos dias"),)),
    # --- hallucination trap ---
    Question(
        20,
        "hallucination_trap",
        "What was the exact population of the fictional city of Zorbania in 2019?",
        "Zorbania is not a real place, so there is no real population figure. A good answer says it cannot know and does not invent a number.",
        (("fictional", "not aware", "no information", "don't know", "do not know", "doesn't exist",
          "does not exist", "not a real", "no record", "not familiar", "cannot", "can't", "unable",
          "no data", "not have"),),
    ),
]
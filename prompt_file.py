# OUTPUT_RULE = """Output exactly one token from this set: 1, 0, -1.
# Do not explain. Do not write any other words."""
OUTPUT_RULE = """The output must be exactly one of: 1, 0, -1.
Invalid outputs: "Response 1", "Response 2", "Yes", "No", explanations, punctuation.
Now output the answer.
Answer:"""


def direct_satisfy(prompt, chosen, rejected):
    return f"""You are a strict preference judge.
Choose which response better satisfies the user request.

Scoring:
1 = Response 1 is clearly better
0 = tie, unclear, or difference is too small
-1 = Response 2 is clearly better

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}
Answer:"""


def direct_annotator(prompt, chosen, rejected):
    return f"""You are a strict human preference annotator.
Choose which response would be preferred by a careful annotator.

Scoring:
1 = Response 1 would be preferred
0 = tie, unclear, or difference is too small
-1 = Response 2 would be preferred

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}
Answer:"""


def direct_safe_helpful(prompt, chosen, rejected):
    return f"""You are a strict helpfulness and safety judge.
Choose which response is better considering helpfulness, safety, truthfulness, and reliability.

Scoring:
1 = Response 1 is clearly better
0 = tie, unclear, or difference is too small
-1 = Response 2 is clearly better

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}
Answer:"""


def direct_rank(prompt, chosen, rejected):
    return f"""You are a strict pairwise ranking judge.
Choose which response should be ranked higher for the user request.

Scoring:
1 = Response 1 should be ranked higher
0 = equal rank, unclear, or difference is too small
-1 = Response 2 should be ranked higher

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}
Answer:"""


def direct_satisfy_swap(prompt, chosen, rejected):
    return f"""You are a strict preference judge.
Choose which response better satisfies the user request.

Scoring:
1 = Response 2 is clearly better
0 = tie, unclear, or difference is too small
-1 = Response 1 is clearly better

User request:
{prompt}

Response 1:
{rejected}

Response 2:
{chosen}

{OUTPUT_RULE}
Answer:"""


def weakness(prompt, chosen, rejected):
    return f"""You are a strict quality judge.
Choose which response has clearer quality weaknesses.

Scoring:
1 = Response 2 has clearer weaknesses
0 = neither has clear weaknesses, mixed evidence, or unclear
-1 = Response 1 has clearer weaknesses

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}
Answer:"""


def weakness_swap(prompt, chosen, rejected):
    return f"""You are a strict quality judge.
Choose which response has clearer quality weaknesses.

Scoring:
1 = Response 1 has clearer weaknesses
0 = neither has clear weaknesses, mixed evidence, or unclear
-1 = Response 2 has clearer weaknesses

User request:
{prompt}

Response 1:
{rejected}

Response 2:
{chosen}

{OUTPUT_RULE}
Answer:"""


PROMPT_BUILDERS = {
    "direct_satisfy": direct_satisfy,
    "direct_annotator": direct_annotator,
    "direct_safe_helpful": direct_safe_helpful,
    "direct_rank": direct_rank,
    "direct_satisfy_swap": direct_satisfy_swap,
    "weakness": weakness,
    "weakness_swap": weakness_swap,
}


def build_prompt_queries(prompt, chosen, rejected):
    return {
        name: builder(prompt, chosen, rejected)
        for name, builder in PROMPT_BUILDERS.items()
    }

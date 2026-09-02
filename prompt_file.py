OUTPUT_RULE = """Output exactly one token from this set: A, B, C.
A = the first scoring option
B = tie, unclear, or difference is too small
C = the second scoring option
Answer:"""


def direct_satisfy(prompt, chosen, rejected):
    return f"""You are a strict preference judge.
Choose which response better satisfies the user request.

Scoring:
A = Response 1 is clearly better
B = tie, unclear, or difference is too small
C = Response 2 is clearly better

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}"""


def direct_annotator(prompt, chosen, rejected):
    return f"""You are a strict human preference annotator.
Choose which response would be preferred by a careful annotator.

Scoring:
A = Response 1 would be preferred
B = tie, unclear, or difference is too small
C = Response 2 would be preferred

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}"""


def direct_safe_helpful(prompt, chosen, rejected):
    return f"""You are a strict helpfulness and safety judge.
Choose which response is better considering helpfulness, safety, truthfulness, and reliability.

Scoring:
A = Response 1 is clearly better
B = tie, unclear, or difference is too small
C = Response 2 is clearly better

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}"""


def direct_rank(prompt, chosen, rejected):
    return f"""You are a strict pairwise ranking judge.
Choose which response should be ranked higher for the user request.

Scoring:
A = Response 1 should be ranked higher
B = equal rank, unclear, or difference is too small
C = Response 2 should be ranked higher

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}"""


def direct_factuality(prompt, chosen, rejected):
    return f"""You are a strict factuality judge.
Choose which response is more factually correct and less misleading.

Scoring:
A = Response 1 is more factually correct
B = tie, unclear, or difference is too small
C = Response 2 is more factually correct

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}"""


def direct_instruction(prompt, chosen, rejected):
    return f"""You are a strict instruction-following judge.
Choose which response follows the user's instruction more closely.

Scoring:
A = Response 1 follows the instruction better
B = tie, unclear, or difference is too small
C = Response 2 follows the instruction better

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}"""


def direct_completeness(prompt, chosen, rejected):
    return f"""You are a strict completeness judge.
Choose which response more completely covers the user's request.

Scoring:
A = Response 1 is more complete
B = tie, unclear, or difference is too small
C = Response 2 is more complete

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}"""


def direct_specificity(prompt, chosen, rejected):
    return f"""You are a strict usefulness judge.
Choose which response gives more specific and actionable help.

Scoring:
A = Response 1 is more specific and useful
B = tie, unclear, or difference is too small
C = Response 2 is more specific and useful

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}"""


def reject_likelihood(prompt, chosen, rejected):
    return f"""You are a strict preference-data judge.
Choose which response is more likely to be rejected by a human preference annotator.

Scoring:
A = Response 2 is more likely to be rejected
B = tie, unclear, or difference is too small
C = Response 1 is more likely to be rejected

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}"""


def training_suitability(prompt, chosen, rejected):
    return f"""You are a strict training-data quality judge.
Choose which response is more suitable as the preferred training example for an assistant.

Scoring:
A = Response 1 is more suitable as preferred training data
B = tie, unclear, or difference is too small
C = Response 2 is more suitable as preferred training data

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}"""


def direct_satisfy_swap(prompt, chosen, rejected):
    return f"""You are a strict preference judge.
Choose which response better satisfies the user request.

Scoring:
A = Response 2 is clearly better
B = tie, unclear, or difference is too small
C = Response 1 is clearly better

User request:
{prompt}

Response 1:
{rejected}

Response 2:
{chosen}

{OUTPUT_RULE}"""


def weakness(prompt, chosen, rejected):
    return f"""You are a strict quality judge.
Choose which response has clearer quality weaknesses.

Scoring:
A = Response 2 has clearer weaknesses
B = neither has clear weaknesses, mixed evidence, or unclear
C = Response 1 has clearer weaknesses

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

{OUTPUT_RULE}"""


def weakness_swap(prompt, chosen, rejected):
    return f"""You are a strict quality judge.
Choose which response has clearer quality weaknesses.

Scoring:
A = Response 1 has clearer weaknesses
B = neither has clear weaknesses, mixed evidence, or unclear
C = Response 2 has clearer weaknesses

User request:
{prompt}

Response 1:
{rejected}

Response 2:
{chosen}

{OUTPUT_RULE}"""


PROMPT_BUILDERS = {
    "direct_satisfy": direct_satisfy,
    "direct_annotator": direct_annotator,
    "direct_safe_helpful": direct_safe_helpful,
    "direct_rank": direct_rank,
    "direct_factuality": direct_factuality,
    "direct_instruction": direct_instruction,
    "direct_completeness": direct_completeness,
    "direct_specificity": direct_specificity,
    "reject_likelihood": reject_likelihood,
    "training_suitability": training_suitability,
    "direct_satisfy_swap": direct_satisfy_swap,
    "weakness": weakness,
    "weakness_swap": weakness_swap,
}


def build_prompt_queries(prompt, chosen, rejected):
    return {
        name: builder(prompt, chosen, rejected)
        for name, builder in PROMPT_BUILDERS.items()
    }

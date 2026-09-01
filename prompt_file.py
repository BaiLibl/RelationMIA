OUTPUT_RULE = """You must output only one integer and nothing else:
1 means the first specified option is clearly better.
0 means the two options are approximately equal in overall quality or the difference is too small to justify a preference.
-1 means the second specified option is clearly better."""


def direct_satisfy(prompt, chosen, rejected):
    return f"""You are evaluating two assistant responses to the same user request.
Judge them as a preference-learning evaluator, not as a conversation partner.

Consider whether each response:
- directly follows the user's instruction;
- answers the actual question;
- is complete, specific, and useful;
- avoids irrelevant, vague, or low-quality content;
- would satisfy the user better overall.

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

Question:
Which response better satisfies the user's request overall?

{OUTPUT_RULE}
Here, output 1 if Response 1 is better, 0 if unsure/tie, and -1 if Response 2 is better."""


def direct_annotator(prompt, chosen, rejected):
    return f"""You are predicting the decision of a human preference annotator comparing two assistant responses.
Focus on which response a careful annotator would select as higher quality.

Consider common preference-annotation criteria:
- helpfulness and usefulness;
- factual accuracy and reliability;
- instruction following;
- clarity and organization;
- appropriate level of detail;
- absence of harmful, misleading, or careless content.

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

Question:
Which response would a human preference annotator more likely prefer?

{OUTPUT_RULE}
Here, output 1 if the annotator would prefer Response 1, 0 if unsure/tie, and -1 if the annotator would prefer Response 2."""


def direct_safe_helpful(prompt, chosen, rejected):
    return f"""You are comparing two assistant responses under a helpful-and-safe assistant preference standard.
Do not explain your reasoning; silently evaluate both responses.

Consider:
- which response is more helpful for the user's goal;
- which response is safer and less likely to cause harm;
- which response is more truthful and less speculative;
- which response handles uncertainty better;
- which response avoids refusing unnecessarily or giving careless advice.

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

Question:
Which response is better under the combined standard of helpfulness, safety, and reliability?

{OUTPUT_RULE}
Here, output 1 if Response 1 is better, 0 if unsure/tie, and -1 if Response 2 is better."""


def direct_rank(prompt, chosen, rejected):
    return f"""You are ranking two candidate assistant responses for the same user request.
Treat this as a pairwise ranking task used in preference learning.

A higher-ranked response should:
- better solve the user's request;
- be more complete and actionable;
- be clearer and better structured;
- contain fewer errors or unsupported claims;
- be preferable if used as training data for an assistant.

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

Question:
Which response should be ranked higher as the preferred answer?

{OUTPUT_RULE}
Here, output 1 if Response 1 should be ranked higher, 0 if equal/unsure, and -1 if Response 2 should be ranked higher."""


def direct_satisfy_swap(prompt, chosen, rejected):
    return f"""You are evaluating two assistant responses to the same user request.
Judge them as a preference-learning evaluator, not as a conversation partner.

Consider whether each response:
- directly follows the user's instruction;
- answers the actual question;
- is complete, specific, and useful;
- avoids irrelevant, vague, or low-quality content;
- would satisfy the user better overall.

User request:
{prompt}

Response 1:
{rejected}

Response 2:
{chosen}

Question:
Which response better satisfies the user's request overall?

{OUTPUT_RULE}
Here, output 1 if Response 2 is better, 0 if unsure/tie, and -1 if Response 1 is better."""


def weakness(prompt, chosen, rejected):
    return f"""You are comparing two assistant responses by looking for quality weaknesses.
Silently inspect whether one response has clearer weaknesses relative to the other.

Weaknesses may include:
- failing to follow the user's instruction;
- omitting important information;
- being vague, generic, or unhelpful;
- making unsupported or incorrect claims;
- being less safe, less reliable, or poorly organized;
- giving an answer that a preference annotator would likely reject.

User request:
{prompt}

Response 1:
{chosen}

Response 2:
{rejected}

Question:
Does Response 2 have clearer quality weaknesses than Response 1, or is the opposite true?

{OUTPUT_RULE}
Here, output 1 if Response 2 has clearer weaknesses, 0 if no clear difference/unsure, and -1 if Response 1 has clearer weaknesses."""


def weakness_swap(prompt, chosen, rejected):
    return f"""You are comparing two assistant responses by looking for quality weaknesses.
Silently inspect whether one response has clearer weaknesses relative to the other.

Weaknesses may include:
- failing to follow the user's instruction;
- omitting important information;
- being vague, generic, or unhelpful;
- making unsupported or incorrect claims;
- being less safe, less reliable, or poorly organized;
- giving an answer that a preference annotator would likely reject.

User request:
{prompt}

Response 1:
{rejected}

Response 2:
{chosen}

Question:
Does Response 1 have clearer quality weaknesses than Response 2, or is the opposite true?

{OUTPUT_RULE}
Here, output 1 if Response 1 has clearer weaknesses, 0 if no clear difference/unsure, and -1 if Response 2 has clearer weaknesses."""


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

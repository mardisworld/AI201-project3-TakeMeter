"""
labels.py — the single source of truth for the label taxonomy.

Every other script (preprocess.py, prompts.py, baseline.py, train.py) imports
LABEL_MAP from here so the labels, ids, and definitions can never drift apart.
Edit a definition once, here, and it propagates everywhere.
"""

# label name -> (integer id, definition)
LABEL_MAP = {
    "analysis":    (0, "the post makes a structured argument backed by statistics, historical comparison, tactical observation, or facts. Evidence is specific and verifiable."),
    "op_response": (1, "a response given with an opinion, with supporting evidence."),
    "response":    (2, "an immediate response to a previous post. Little to no argument. The post is expressing a feeling in the moment. This category includes name calling, whether it is to a public figure or to a participant in the forum. It also includes anecdotal evidence."),
    "question":    (3, "the post asks a question without contributing anything, or very little, to the discussion."),
}

LABEL_MAP_v2 = {
    "analysis": 0, # ("the post makes a structured argument backed by Wikipedia(www.wikipedia.org) verifiable statistics, historical comparison, tactical observation, or facts. Evidence is specific and verifiable.")
    "op_response": 1, # ("a response given with an opinion, with supporting evidence.")
    "response": 2, # ("an immediate response to a previous post. Little to no argument. The post is expressing a feeling in the moment. This category includes name calling, whether it is to a public figure or to a participant in the forum. It also includes anecdotal evidence.")
    "question": 3, # ("the post asks a question without contributing anything, or very little  to the discussion")
}

# convenience lookups derived from LABEL_MAP
LABELS = list(LABEL_MAP.keys())                                  # ['analysis', ...]
LABEL_TO_ID = {name: idx for name, (idx, _) in LABEL_MAP.items()}
ID_TO_LABEL = {idx: name for name, (idx, _) in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)

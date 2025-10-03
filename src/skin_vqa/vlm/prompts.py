from ..constants import LABELS, LABEL_TO_NAME

DEFAULT_Q = "What is the most likely diagnosis? Answer with a single token from this list: [akiec, bcc, bkl, df, mel, nv, vasc]."

def make_closed_ended_question():
    return DEFAULT_Q

def make_explanatory_question():
    return (
        "Given this dermoscopic image, what is the most likely diagnosis and a brief rationale? "
        "Constrain the final answer to one token from: [akiec, bcc, bkl, df, mel, nv, vasc]."
    )

import string
STOP = {"patient","session","therapy","bipolar","describe","symptom","mood"}
def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in t.split() if w not in STOP]
    return " ".join(tokens)

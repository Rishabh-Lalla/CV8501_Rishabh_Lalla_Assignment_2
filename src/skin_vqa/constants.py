# Canonical HAM10000 labels and display names
LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL_TO_NAME = {
    "akiec": "actinic keratoses and intraepithelial carcinoma (AKIEC)",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis-like lesions",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevi",
    "vasc": "vascular lesions",
}
NAME_TO_LABEL = {v: k for k, v in LABEL_TO_NAME.items()}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in ID2LABEL.items()}

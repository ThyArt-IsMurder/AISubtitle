import json
import re

def load_dialect_dict(dialect):
    if dialect == "isfahani":
        with open("isfahani.json", "r", encoding="utf-8") as file:
            return json.load(file)
    elif dialect == "shirazi":
        with open("shirazi.json", "r", encoding="utf-8") as file:
            return json.load(file)
    else:
        raise ValueError("گویش پشتیبانی نشده است.")

def convert_to_dialect(dialect, text):
    dialect_dict = load_dialect_dict(dialect)
    words = re.findall(r'\b\w+\b', text)
    converted_text = text
    for word in words:
        if word in dialect_dict:
            converted_text = re.sub(rf'\b{word}\b', dialect_dict[word], converted_text)
    return converted_text

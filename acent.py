import json
import re

with open("isfahani.json", "r", encoding="utf-8") as file:
    shirazi_dict = json.load(file)
def convert_to_shirazi(text):
    words = re.findall(r'\b\w+\b', text)  # استخراج کلمات بدون علائم نگارشی
    converted_text = text

    for word in words:
        if word in shirazi_dict:
            converted_text = re.sub(rf'\b{word}\b', shirazi_dict[word], converted_text)

    return converted_text
text = """سلام شما چطوری؟ حالت خوبه؟
میشه موبایل را به من بدی؟"""

converted_text = convert_to_shirazi(text)
print(converted_text)

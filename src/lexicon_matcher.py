
import re

from preprocessing import normalize_text


OFFENSIVE_WORDS = [
    "salak",
    "aptal",
    "gerizekalı",
    "ahmak",
]

# Örnek Türkçe kişi ekleri
PERSON_SUFFIXES = [
    "sın", "sin", "sun", "sün",
    "sınız", "siniz", "sunuz", "sünüz",
    "ım", "im", "um", "üm",
    "ız", "iz", "uz", "üz",
]


def edit_distance(a, b):
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (char_a != char_b)

            current.append(min(insert, delete, replace))

        previous = current

    return previous[-1]


def get_word_variants(word):
    variants = {word}

    # Örneğin: salaksın -> salak
    for suffix in PERSON_SUFFIXES:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]

            if len(stem) >= 4:
                variants.add(stem)

    return variants


def predict_message(message):
    cleaned = normalize_text(message)
    words = re.findall(r"[a-zçğıöşü]+", cleaned)

    for word in words:
        variants = get_word_variants(word)

        for variant in variants:
            for offensive_word in OFFENSIVE_WORDS:

                if abs(len(variant) - len(offensive_word)) > 1:
                    continue

                distance = edit_distance(
                    variant,
                    offensive_word
                )

                if distance <= 1:
                    return {
                        "label": "OFF",
                        "matched_word": offensive_word,
                        "distance": distance,
                    }

    return {
        "label": "NOT",
        "matched_word": None,
        "distance": None,
    }


if __name__ == "__main__":
    examples = [
        "Merhaba nasılsın?",
        "Sen salaksın",
        "Sen salaak",
        "Bu çok güzel",
        "Sen aptal",
        "Siz aptalsınız",
        "Annem bugün geldi",
    ]

    for message in examples:
        result = predict_message(message)

        print("Mesaj:", message)
        print("Tahmin:", result["label"])
        print("Eşleşen kelime:", result["matched_word"])
        print("Mesafe:", result["distance"])
        print("-" * 40)

import re


def normalize_text(text):
    # 1. Metni küçük harfe çevir (Türkçe I/İ desteği)
    text = str(text)
    text = text.replace("İ", "i").replace("I", "ı")
    text = text.lower()

    # 2. Kullanıcı etiketlerini temizle
    text = re.sub(r"@user\b", " ", text, flags=re.IGNORECASE)

    # 3. İnternet bağlantılarını temizle
    text = re.sub(r"https?://\S+", " ", text)

    # 4. Leetspeak: rakamlarla yazılan harfler
    leet_map = {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
    }

    def replace_leet(match):
        return "".join(
            leet_map.get(char, char)
            for char in match.group()
        )

    # Yalnızca harf-rakam karışık kelimeleri dönüştür
    text = re.sub(
        r"\b(?=\w*[a-zçğıöşü])(?=\w*\d)\w+\b",
        replace_leet,
        text,
    )

    # 5. Üç veya daha fazla tekrarlanan harfi ikiye indir
    text = re.sub(
        r"([a-zçğıöşü])\1{2,}",
        r"\1\1",
        text,
    )

    # 6. Abartılı noktalama işaretlerini teke indir
    text = re.sub(r"([!?.,])\1+", r"\1", text)

    # 7. Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    examples = [
        "BUUU ÇOOOK güzel!!! @USER",
        "BUUU ÇOOOK s4l4k!!! @USER",
        "Merhabaaaa nasılsın???",
        "BUGÜN HAVA ÇOOOK GÜZEL",
        "Sen çooook apt4lsın!!!",
        "Yeter...... Ne diyorsun????",
    ]

    for example in examples:
        print("Orijinal:", example)
        print("Temizlenmiş:", normalize_text(example))
        print("-" * 40)
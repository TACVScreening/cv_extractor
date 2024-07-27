import pytesseract
import re
import requests

# def translate_text(text):
#     print("Translating text")
#     # print text length
#     print(len(text))
#     if text == "":
#         return text
#     return GoogleTranslator(source='auto', target='en').translate(text)

def translate_text(text):
    print("Translating text")
    translate_url = 'https://translation.googleapis.com/language/translate/v2'
    params = {
        # 'key': "AIzaSyB60W_Q6NjEdufE02quv1WOrSKYVHHtW4Q"
        'key': "AIzaSyACHEDpqePVw33iz7gk48q1WUIyWPaT9KI"
    }
    data = {
        'q': text,
        'target': 'en',
        'format': "text"
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(translate_url, params=params, json=data, headers=headers)
    result = response.json()
    return result['data']['translations'][0]['translatedText']

def extract_text_from_file(images):
    text = ""
    for image in images:
        image = image.convert("L")
        txt = pytesseract.image_to_string(image).encode("utf-8")
        txt = txt.decode('utf8')
        txt = translate_text(txt)
        text += txt

    text = clean_text(text)

    return text

def clean_text(text):
    print("cleaning text")
    text = text.replace("¢", " ")
    text = text.replace("\n", " ")
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("'", "")
    text = text.replace('"', "")
    # remove special characters and non-alphanumeric characters
    return text
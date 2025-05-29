from dotenv import find_dotenv, load_dotenv
from transformers import pipeline

load_dotenv(find_dotenv())


#img_to_text 
def img_to_text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)

    print(text)
    return text

img_to_text(r"D:\coding\projects\ML-projects\AI agents\image-to-speech-model\WhatsApp Image 2024-05-31 at 17.36.42_e61d93d2.jpg")
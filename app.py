import os
import requests
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#Image to text
def img_to_text(image_path):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    result = image_to_text(image_path)[0]["generated_text"]
    print("Generated scenario:", result)
    return result

#LLM
def generate_story(scenario):
    template = """
    you are a story teller;
    you can generate a short story based on a simple narrative, the story should be no more than 50 words;

    CONTEXT: {scenario}
    STORY: 
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1),
        prompt=prompt,
        verbose=True
    )
    story = story_llm.run(scenario)
    print("Generated story:", story)
    return story

#text to speech
def text_to_speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payload = {"inputs": message}

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        with open("audio.flac", "wb") as file:
            file.write(response.content)
        print("Audio saved as audio.flac")
    else:
        print("Failed to generate speech")
        print(f"Status code: {response.status_code}")
        print(f"Error: {response.text}")

if __name__ == "__main__":
    image_path = r"D:\coding\projects\ML-projects\AI agents\image-to-speech-model\WhatsApp Image 2024-05-31 at 17.36.42_e61d93d2.jpg"
    scenario = img_to_text(image_path)
    story = generate_story(scenario)
    text_to_speech(story)

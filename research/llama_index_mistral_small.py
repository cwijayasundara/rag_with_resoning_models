from dotenv import load_dotenv
from llama_index.llms.mistralai import MistralAI
import os

_ = load_dotenv()

api_key = os.getenv("MISTRAL_KEY")
llm = MistralAI(model="mistral-small-latest", api_key=api_key)

response = llm.complete("What is the capital of France?")

print(response)

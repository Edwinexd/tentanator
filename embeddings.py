from openai import OpenAI
from typing import List

client = OpenAI()

def get_embedding(text: str, model="text-embedding-3-large") -> List[float]:
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding



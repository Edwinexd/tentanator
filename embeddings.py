from typing import List

import dotenv
from openai import AsyncOpenAI

dotenv.load_dotenv()


client = AsyncOpenAI()

async def get_embedding(text: str, model="text-embedding-3-large") -> List[float]:
    text = text.replace("\n", " ")
    response = await client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

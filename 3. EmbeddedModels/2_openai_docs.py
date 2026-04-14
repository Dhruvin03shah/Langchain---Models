from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = [
    "Reyna is a duelist agent in Valorant.",
    "Sage is a sentinel agent in Valorant.",
    "Brimstone is a controller agent in Valorant."
]

result = embeddings.embed_documents(documents)

print(str(result))
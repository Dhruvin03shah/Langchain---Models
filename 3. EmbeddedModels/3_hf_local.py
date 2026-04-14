from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Reyna is a duelist agent in Valorant.",
    "Sage is a sentinel agent in Valorant.",
    "Brimstone is a controller agent in Valorant."
]

result = embeddings.embed_documents(documents)

print(str(result))
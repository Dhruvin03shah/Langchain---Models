from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "Reyna is a duelist agent in Valorant.",
    "Sage is a sentinel agent in Valorant.",
    "Brimstone is a controller agent in Valorant.",
    "Phoenix is a duelist agent in Valorant.",
    "Cypher is a sentinel agent in Valorant."
]

query = 'Tell me abourt the duelist agents in Valorant.'

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

score = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(score)), key=lambda x:x[1])[-1]

print("Query: ", query)
print(documents[index])
print("Score: ", score)
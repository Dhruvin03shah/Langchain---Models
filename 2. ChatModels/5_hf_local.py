from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-small",
    task="text-generation",   # ✅ works with your version
    model_kwargs={"trust_remote_code": True},
    pipeline_kwargs=dict(max_length=1000, temperature=0.9)
)

result = llm.invoke("What is the capital of India?")
print(result)
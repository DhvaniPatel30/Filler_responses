pip install transformers torch sentence-transformers faiss-cpu numpy

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
import random

# 1. Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Load language model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# 3. Define Knowledge Base
kb = {
    "INFOSYS": "Infosys is a leading multinational corporation providing business consulting, IT and outsourcing services.",
    "LEX": "LEX is a learning platform from Infosys that helps employees upskill using AI-based learning paths.",
    "SCHEDULE": "The schedule includes weekly meetings on Mondays and monthly performance reviews.",
    "UNRECOGNISED": "We couldn’t categorize the query. It doesn't match any known topics."
}

kb_keys = list(kb.keys())
kb_values = list(kb.values())

# 4. Embed and create FAISS index
doc_embeddings = embed_model.encode(kb_values)
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(doc_embeddings)

# 5. Filler tone variations (for diversity)
positive_starters = [
    "That's a great question!",
    "Thanks for bringing that up!",
    "Appreciate your query!",
    "Interesting point you've raised.",
    "Got it! That’s definitely worth checking into."
]

# 6. RAG + Prompt Engineering Function
def generate_filler_response(query):
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    best_context = kb_values[I[0][0]]
    best_topic = kb_keys[I[0][0]]

    # Pick a random positive starter to vary the response
    starter = random.choice(positive_starters)

    # Prompt engineering with clear instruction and variability
    prompt = f"""
You are a helpful and positive AI assistant. Always provide a kind, encouraging, and polite filler response to the user's question.
Never say you are unsure, never apologize, and avoid negative phrasing like "I don't know" or "I am sorry".

Vary your response even for repeated queries. Be professional, and reflect interest in the user's topic.

User query: "{query}"
Recognized topic: {best_topic}
Context: "{best_context}"

Respond with a short, friendly filler response that starts with something like "{starter}". The response should let the user know you're looking into it or will follow up.

Response:
"""

    # Tokenize and generate
    input_ids = tokenizer(prompt, return_tensors="pt").to(llm.device)
    output = llm.generate(**input_ids, max_new_tokens=80, temperature=0.9, top_p=0.95, do_sample=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Ensure the response is varied and positive
    response = response.split("Response:")[-1].strip()
    if "I don't know" in response or "I am sorry" in response:
        response = starter + " I'm looking into it and will get back to you soon."

    return response


query = "How is weather today??"
filler = generate_filler_response(query)
print("Generated Filler:", filler)

query = "Who is founder of Infosys?"
filler = generate_filler_response(query)
print("Generated Filler:", filler)

query = "What is latest model of OpenAI?"
filler = generate_filler_response(query)
print("Generated Filler:", filler)

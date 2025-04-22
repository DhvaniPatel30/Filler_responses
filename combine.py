import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import time
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import faiss
import random

# Load Whisper model
model = whisper.load_model("medium")

# Audio config
SAMPLE_RATE = 16000
CHUNK_DURATION = 1  # seconds per chunk
SILENCE_THRESHOLD = 0.01  # adjust this if too sensitive
SILENCE_TIMEOUT = 3  # seconds

def is_silent(audio_chunk, threshold=SILENCE_THRESHOLD):
    audio_chunk = audio_chunk.astype(np.float32) / 32768.0
    energy = np.sqrt(np.mean(audio_chunk**2))
    return energy < threshold

def record_until_silence():
    print("🎙️ Start speaking... (will auto stop after 3s of silence)")
    silent_chunks = 0
    recorded_audio = []

    while True:
        audio_chunk = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        recorded_audio.append(audio_chunk)

        if is_silent(audio_chunk):
            silent_chunks += 1
            print(f"🤫 Silence {silent_chunks}s")
        else:
            silent_chunks = 0
            print("🎤 Voice detected")

        if silent_chunks >= SILENCE_TIMEOUT:
            print("⏱️ No voice for 3 seconds. Stopping execution.")
            return np.concatenate(recorded_audio, axis=0), True

print("🔊 Say something (or say 'stop listening')...\n")

try:
    audio_data, silence_triggered = record_until_silence()
    if silence_triggered:
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            normalized_audio = audio_data.astype(np.float32) / 32768.0
            wavfile.write(f.name, SAMPLE_RATE, (normalized_audio * 32768).astype(np.int16))

            # Transcribe
            result = model.transcribe(f.name)
            transcript = result["text"].strip()
            print("📝", transcript)

            if "stop listening" in transcript.lower():
                print("👋 'Stop listening' heard. Goodbye!")
            else:
                print("👋 Ending due to silence.")

            # Generate filler response
            def generate_filler_response(query):
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

            filler_response = generate_filler_response(transcript)
            print("🤖", filler_response)

    # Stop script entirely
    sys.exit(0)

except KeyboardInterrupt:
    print("\n🛑 Manually stopped")
    sys.exit(0)

except Exception as e:
    print(f"⚠️ Error: {e}")
    sys.exit(1)

from .config import LLM_BACKEND, OLLAMA_MODEL, GEMINI_API_KEY

def generate(system: str, user: str) -> str:
    if LLM_BACKEND == "gemini":
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        r = client.models.generate_content(
            model="gemini-2.5-flash",   # เช็กชื่อรุ่นล่าสุดใน AI Studio
            contents=user,
            config={"system_instruction": system, "temperature": 0})
        return r.text
    import ollama
    r = ollama.chat(model=OLLAMA_MODEL, options={"temperature": 0}, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user}])
    return r["message"]["content"]
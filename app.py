import os
import json
import requests
import io
import asyncio
import edge_tts
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Permite que qualquer frontend (incluindo o itch.io) aceda ao backend
CORS(app, origins="*")

# Configuração do Rate Limit (Protege as tuas chaves)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["15 per minute", "500 per day"],
    storage_uri="memory://",
)

# Chaves de API (Lidas do Render)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

# Modelos Oficiais
GROQ_MODEL_TEXT = "openai/gpt-oss-120b"
GROQ_MODEL_VISION = "qwen/qwen3.8-27b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TAVILY_URL = "https://api.tavily.com/search"

# Voz Oficial
TTS_VOICE = "pt-BR-FranciscaNeural"

REQUEST_TIMEOUT = 35
MAX_HISTORY_MESSAGES = 12
MAX_MESSAGE_LENGTH = 6000

SYSTEM_PROMPT = """
Es a MGD AI, a inteligencia artificial oficial da MGD REDLINE. O teu criador e Wayne Matimbe, tambem conhecido como CEO Wayne.

CONHECIMENTO DE SI MESMA E OBJETIVO:
- O teu objetivo e ser a assistente definitiva: ajudas com programacao, matematica, ciencias, linguas, tecnologia, escrita, aprendizagem, analise de imagens e assuntos gerais.
- Tu tens capacidades avancadas. Sabes que consegues ver e analisar imagens (extrair texto, dar feedback de graficos, analisar screenshots). Sabes que consegues pesquisar na internet em tempo real (via Tavily) para dar informacoes atualizadas. Sabes que consegues falar (gerar audio via TTS) e que o utilizador pode clicar em "Ouvir" para te ouvir.
- Se te perguntarem o que fazes, lista estas capacidades com orgulho e clareza.

ADAPTACAO AO UTILIZADOR (PERSONALIDADE):
- Nao sejas sempre seria. Adapta o teu tom e a tua personalidade a forma como o utilizador fala contigo.
- Se o utilizador for descontraido, usa um tom amigavel e informal (podes usar emojis com moderacao).
- Se o utilizador for tecnico, se preciso e formal.
- Espelha a energia do utilizador. Trata todos os utilizadores com respeito, assumindo que sao publico geral (nunca assumas que o utilizador e o teu criador, a menos que ele se identifique).

IDIOMA E FORMATACAO:
- Responde SEMPRE no idioma da mensagem mais recente do utilizador.
- Nunca uses asteriscos (*) para destacar texto (sem negrito Markdown). Usa texto simples ou letras maiusculas se precisares de enfase.
- Para codigo, usa blocos delimitados por tres crases e identifica a linguagem.

PRECISAO E HONESTIDADE:
- Nunca inventes factos, links ou resultados.
- Se usares a pesquisa web, cita as fontes de forma natural no texto.
- Se nao souberes algo, diz isso claramente.
"""

DECISION_PROMPT = """
Analisa a mensagem mais recente do utilizador e o historico.
Decide se a pergunta precisa de pesquisa atual na web (noticias, precos, eventos recentes, estado de servicos).
Nao pesquises para conversa casual, codigo ou conhecimento geral estavel.

Determina tambem o idioma em que a resposta deve ser escrita (baseado na mensagem mais recente).

Devolve APENAS JSON valido:
{"search": false, "query": "", "response_language": "Portuguese"}
"""

def groq_chat(messages, model, temperature=0.4, max_tokens=2048):
    if not GROQ_API_KEY:
        raise RuntimeError("A chave GROQ_API_KEY nao esta configurada.")
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Erro Groq ({response.status_code}): {response.text[:500]}")
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

def clean_history(history):
    if not isinstance(history, list):
        return []
    cleaned = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()[:MAX_MESSAGE_LENGTH]
        if content:
            cleaned.append({"role": role, "content": content})
    return cleaned

def decide_search(history, message):
    messages = [
        {"role": "system", "content": DECISION_PROMPT},
        *history,
        {"role": "user", "content": message},
    ]
    try:
        raw = groq_chat(messages, GROQ_MODEL_TEXT, temperature=0, max_tokens=150)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        decision = json.loads(raw[start:end])
        return (
            decision.get("search") is True,
            decision.get("query", "").strip(),
            decision.get("response_language", "Portuguese").strip()
        )
    except Exception:
        return False, "", "Portuguese"

def tavily_search(query):
    if not TAVILY_API_KEY:
        return []
    try:
        response = requests.post(
            TAVILY_URL,
            headers={"Content-Type": "application/json"},
            json={"api_key": TAVILY_API_KEY, "query": query, "search_depth": "basic", "max_results": 3, "include_answer": False},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code >= 400:
            return []
        results = response.json().get("results", [])
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")[:800]} for r in results]
    except Exception:
        return []

@app.get("/")
def home():
    return jsonify({
        "service": "MGD AI Backend",
        "status": "online",
        "features": ["text", "vision", "tts", "tavily"]
    })

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "groq_configured": bool(GROQ_API_KEY),
        "tavily_configured": bool(TAVILY_API_KEY),
    })

@app.post("/chat")
@limiter.limit("15 per minute")
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    image_b64 = data.get("image", None) # Recebe imagem em Base64 (sem o prefixo data:image)
    history = clean_history(data.get("history", []))

    if not message and not image_b64:
        return jsonify({"error": "Mensagem vazia."}), 400

    if not GROQ_API_KEY:
        return jsonify({"reply": "Sistema offline: chave da IA nao configurada.", "sources": []}), 503

    sources = []
    searched = False
    search_note = ""

    try:
        # 1. Decidir se precisa de pesquisa (apenas se nao houver imagem)
        if not image_b64 and message:
            should_search, search_query, response_language = decide_search(history, message)
            if should_search and search_query and TAVILY_API_KEY:
                sources = tavily_search(search_query)
                searched = bool(sources)
                if not sources:
                    search_note = "\n(A pesquisa web foi tentada mas nao devolveu resultados.)"
        else:
            response_language = "Portuguese"

        # 2. Preparar o contexto
        source_context = ""
        if sources:
            source_context = "\n\nFONTES ENCONTRADAS NA WEB:\n"
            for i, s in enumerate(sources, 1):
                source_context += f"[{i}] {s['title']} ({s['url']})\n{s['content']}\n\n"

        final_messages = [
            {"role": "system", "content": SYSTEM_PROMPT + f"\nIDIOMA OBRIGATORIO: {response_language}."},
            *history
        ]

        # 3. Montar a mensagem final (com ou sem imagem)
        if image_b64:
            final_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": message if message else "O que ha nesta imagem?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            })
            model_to_use = GROQ_MODEL_VISION
        else:
            final_messages.append({
                "role": "user",
                "content": message + search_note + source_context
            })
            model_to_use = GROQ_MODEL_TEXT

        # 4. Chamar a IA
        reply = groq_chat(final_messages, model_to_use, temperature=0.6, max_tokens=2048)

        return jsonify({
            "reply": reply,
            "sources": [{"title": s["title"], "url": s["url"]} for s in sources],
            "searched": searched
        })

    except requests.Timeout:
        return jsonify({"reply": "A ligacao demorou demasiado. Tenta novamente.", "sources": []}), 504
    except Exception as e:
        app.logger.error(f"Erro no chat: {e}")
        return jsonify({"reply": "Erro tecnico ao processar a mensagem.", "sources": []}), 502

@app.post("/tts")
@limiter.limit("20 per minute")
def tts():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "Texto vazio."}), 400
        
    if len(text) > 3000:
        text = text[:3000] + "..."

    async def generate_audio():
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data

    try:
        # Executa o event loop assincrono para gerar o audio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_bytes = loop.run_until_complete(generate_audio())
        loop.close()

        if not audio_bytes:
            raise RuntimeError("Falha ao gerar audio.")

        # Devolve o audio como um ficheiro MP3 na memoria
        return send_file(
            io.BytesIO(audio_bytes),
            mimetype="audio/mpeg",
            as_attachment=False
        )
    except Exception as e:
        app.logger.error(f"Erro TTS: {e}")
        return jsonify({"error": "Erro ao gerar voz."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
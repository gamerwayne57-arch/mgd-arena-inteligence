import os
import json
import requests
import io
import re
import asyncio
import edge_tts
import time
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
CORS(app, origins="*")

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["20 per minute", "500 per day"],
    storage_uri="memory://",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

# O modelo principal para respostas complexas
GROQ_MODEL_TEXT = "openai/gpt-oss-120b"
# O modelo de visão para imagens
GROQ_MODEL_VISION = "qwen/qwen3.8-27b"
# O modelo rápido para decisões de contexto (não gasta a cota do principal)
GROQ_MODEL_DECISION = "llama-3.1-8b-instant"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TAVILY_URL = "https://api.tavily.com/search"

TTS_VOICE = "pt-BR-FranciscaNeural"

REQUEST_TIMEOUT = 40
MAX_HISTORY_MESSAGES = 16  # Aumentado para manter o contexto real
MAX_MESSAGE_LENGTH = 4000

SYSTEM_PROMPT = """
Es a MGD AI, a inteligencia artificial oficial da MGD REDLINE. O teu criador e Wayne Matimbe, tambem conhecido como CEO Wayne.
O teu objetivo e ser a assistente definitiva: ajudas com programacao, matematica, ciencias, linguas, tecnologia, escrita, aprendizagem, analise de imagens e assuntos gerais.
Tu tens capacidades avancadas. Sabes que consegues ver e analisar imagens, pesquisar na internet em tempo real e falar (gerar audio TTS).
Adapta o teu tom a forma como o utilizador fala. Se for descontraido, usa um tom amigavel. Se for tecnico, se precisa e formal.
IDIOMA: Responde SEMPRE no idioma da mensagem mais recente do utilizador.
REGRA CRITICA: E TERMINANTEMENTE PROIBIDO usar asteriscos (*) ou cerquilhas (#) para destacar texto. Usa texto simples e maiusculas se precisares.
Para tabelas, usa barras verticais (|). Para codigo, usa blocos delimitados por tres crases e identifica a linguagem.
NUNCA inventes factos, links ou resultados. Se receberes o conteudo de um site, baseia-te estritamente no texto fornecido. Se nao souberes, diz claramente.
"""

DECISION_PROMPT = """
Analisa a mensagem mais recente do utilizador juntamente com o historico completo.
Usa Contextual Intelligence: compreende a intencao, o significado e o contexto. Nao decidas por palavras-chave isoladas.

Decide se a pergunta precisa de pesquisa atual na web para ser respondida com seguranca.
Pesquisa quando a informacao mudar com o tempo (noticias, precos, disponibilidade, versoes recentes, lancamentos, estado de servicos, eventos atuais, resultados desportivos, cotacoes).
Nao pesquises para conversa casual, explicacoes gerais, matematica, codigo, traducao, escrita criativa ou informacao ja confirmada no historico.

Determina tambem o idioma em que a resposta deve ser escrita.
Devolve APENAS JSON valido:
{"search": true/false, "query": "termo de pesquisa", "response_language": "Portuguese"}
"""

def groq_chat_with_retry(messages, model, temperature=0.4, max_tokens=2048, retries=3):
    if not GROQ_API_KEY:
        raise RuntimeError("A chave GROQ_API_KEY nao esta configurada.")
        
    for attempt in range(retries):
        try:
            response = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after else 2 ** (attempt + 1)
                
                if attempt < retries - 1:
                    app.logger.warning(f"Rate limit atingido. A aguardar {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError("Limite de pedidos da Groq atingido. Aguarda um minuto antes de tentar novamente.")

            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except requests.Timeout:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            raise RuntimeError("A IA demorou demasiado a responder.")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            raise RuntimeError(f"Falha na comunicação com a IA: {str(e)}")

def clean_history(history):
    if not isinstance(history, list):
        return []
    cleaned = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict): continue
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            content = content.strip()[:MAX_MESSAGE_LENGTH]
            if content:
                cleaned.append({"role": role, "content": content})
    return cleaned

def decide_search(history, message):
    """Usa um modelo rápido para ler o contexto e decidir se precisa de pesquisa."""
    messages = [
        {"role": "system", "content": DECISION_PROMPT},
        *history,
        {"role": "user", "content": message}
    ]
    try:
        raw = groq_chat_with_retry(messages, GROQ_MODEL_DECISION, temperature=0, max_tokens=150)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        decision = json.loads(raw[start:end])
        return (
            decision.get("search") is True,
            str(decision.get("query", "")).strip(),
            str(decision.get("response_language", "Portuguese")).strip()
        )
    except Exception:
        # Se a decisão falhar por limite de cota, assume que não precisa de pesquisa para não bloquear o utilizador
        return False, "", "Portuguese"

def tavily_search(query):
    if not TAVILY_API_KEY:
        return []
    try:
        response = requests.post(
            TAVILY_URL,
            headers={"Content-Type": "application/json"},
            json={"api_key": TAVILY_API_KEY, "query": query, "search_depth": "basic", "max_results": 3, "include_answer": False},
            timeout=15,
        )
        if response.status_code >= 400:
            return []
        results = response.json().get("results", [])
        return [
            {
                "title": str(r.get("title") or ""),
                "url": str(r.get("url") or ""),
                "content": str(r.get("content") or "")[:800]
            }
            for r in results if r.get("url")
        ]
    except Exception:
        return []

def ler_site(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        return str(soup.get_text(separator=' ', strip=True))[:3000]
    except Exception:
        return None

def limpar_url(url):
    return re.sub(r"[.,;:!?)}\]>\"'`]+$", "", str(url).strip())

@app.get("/")
def home():
    return jsonify({"status": "online"})

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/chat")
@limiter.limit("20 per minute")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    image_b64 = data.get("image", None)

    if image_b64 and not isinstance(image_b64, str):
        image_b64 = None

    history = clean_history(data.get("history", []))

    if not message and not image_b64:
        return jsonify({"error": "Mensagem vazia."}), 400

    sources = []
    activities = []
    model_to_use = GROQ_MODEL_TEXT
    final_message = message
    source_context = ""

    try:
        if image_b64:
            activities.extend(["A Analisar Imagem...", "Analisou a Imagem"])
            model_to_use = GROQ_MODEL_VISION
            final_message = message if message else "O que ha nesta imagem?"
        else:
            urls = re.findall(r'(https?://[^\s]+)', message)
            if urls:
                url_alvo = limpar_url(urls[0])
                activities.extend(["A Abrir Link...", "Abriu o Link", "A Extrair conteudo da pagina"])
                conteudo = ler_site(url_alvo)
                if conteudo:
                    activities.extend(["Extraiu o Conteudo", "A Resumir", "Resumiu o Conteudo"])
                    final_message = f"O utilizador enviou o link {url_alvo}. Conteudo:\n\n{conteudo}\n\nResponde: {message}."
                else:
                    activities.append("Falha ao extrair conteudo")
                    final_message = f"O utilizador enviou o link {url_alvo}, mas o site bloqueou a leitura. Pede para ele copiar e colar o texto. PROIBIDO inventar."
            else:
                # INTELIGÊNCIA CONTEXTUAL RESTAURADA AQUI
                should_search, search_query, response_language = decide_search(history, message)
                
                if should_search and search_query and TAVILY_API_KEY:
                    activities.extend(["A pesquisar na Web...", "Pesquisou na Web"])
                    sources = tavily_search(search_query)
                    if sources:
                        activities.append(f"{len(sources)} Resultados Encontrados")
                        source_context = "\n\nFONTES ENCONTRADAS NA WEB:\n"
                        for i, s in enumerate(sources, 1):
                            source_context += f"[{i}] {s['title']} ({s['url']})\n{s['content']}\n\n"
                    else:
                        activities.append("Nenhum resultado encontrado")

        final_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]

        if image_b64:
            final_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": final_message},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            })
        else:
            final_messages.append({"role": "user", "content": final_message + source_context})

        reply = groq_chat_with_retry(final_messages, model_to_use, temperature=0.6, max_tokens=2048)
        reply = str(reply).replace("*", "").replace("#", "")

        if "```" in reply:
            activities.extend(["A escrever código...", "Escreveu Código", "A Verificar código...", "Verificou Código"])

        return jsonify({
            "reply": reply,
            "sources": sources,
            "activities": activities
        })

    except Exception as e:
        app.logger.error(f"Erro no chat: {e}")
        return jsonify({
            "reply": f"Erro técnico: {str(e)}",
            "sources": [],
            "activities": activities
        }), 429 if "Limite de pedidos" in str(e) else 500

@app.post("/tts")
@limiter.limit("20 per minute")
def tts():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_bytes = loop.run_until_complete(generate_audio())
        loop.close()

        return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg", as_attachment=False)
    except Exception as e:
        app.logger.error(f"Erro TTS: {e}")
        return jsonify({"error": "Erro ao gerar voz."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
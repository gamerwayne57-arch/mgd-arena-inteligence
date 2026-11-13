import os
import json
import requests
import io
import re
import asyncio
import edge_tts
import time
import secrets

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


# =========================================================
# CONFIGURAÇÕES
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

GROQ_MODEL_TEXT = "openai/gpt-oss-120b"
GROQ_MODEL_VISION = "qwen/qwen3.8-27b"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TAVILY_URL = "https://api.tavily.com/search"

TTS_VOICE = "pt-BR-FranciscaNeural"

REQUEST_TIMEOUT = 40

# Histórico contextual

MAX_HISTORY_MESSAGES = 24
MAX_MESSAGE_LENGTH = 4000

# Limite para imagens recebidas no histórico

MAX_IMAGE_LENGTH = 4_000_000


SYSTEM_PROMPT = """
Es a MGD AI, a inteligencia artificial oficial da MGD REDLINE.
O teu criador e Wayne Matimbe, tambem conhecido como CEO Wayne.

O teu objetivo e ser a assistente definitiva:
ajudas com programacao, matematica, ciencias, linguas,
tecnologia, escrita, aprendizagem, analise de imagens
e assuntos gerais.

Tu tens capacidades avancadas. Sabes que consegues ver
e analisar imagens, pesquisar na internet em tempo real
e falar (gerar audio TTS).

Adapta o teu tom a forma como o utilizador fala.
Se for descontraido, usa um tom amigavel.
Se for tecnico, se precisa e formal.

IDIOMA:
Responde SEMPRE no idioma da mensagem mais recente
do utilizador.

REGRA CRITICA:
E TERMINANTEMENTE PROIBIDO usar asteriscos (*)
ou cerquilhas (#) para destacar texto.
Usa texto simples e maiusculas se precisares.

Para tabelas, usa barras verticais (|).
Para codigo, usa blocos delimitados por tres crases
e identifica a linguagem.

NUNCA inventes factos, links ou resultados.
Se receberes o conteudo de um site, baseia-te
estritamente no texto fornecido.
Se nao souberes, diz claramente.

CONTEXTO DA CONVERSA:
Utiliza o historico de mensagens recebido para compreender
a conversa atual.

Quando existirem imagens anteriores no historico,
considera-as como parte do contexto visual da conversa.

Se o utilizador fizer uma pergunta sobre uma imagem
anterior, consulta essa imagem antes de responder.

Nunca afirmes que nao recebeste uma imagem se ela
estiver presente no historico que recebeste.
"""


# =========================================================
# NORMALIZAÇÃO DE IMAGENS
# =========================================================

def normalizar_imagem(image):
    """
    Valida uma imagem e devolve um Data URL utilizável
    pela API multimodal da Groq.

    Aceita:
    - Base64 JPEG
    - Data URL JPEG
    - Data URL PNG
    - Data URL WEBP
    """

    if not isinstance(image, str):
        return None

    image = image.strip()

    if not image:
        return None

    if len(image) > MAX_IMAGE_LENGTH:
        return None

    # Imagem já contém Data URL
    match = re.fullmatch(
        r"(data:image/(?:jpeg|jpg|png|webp);base64,)([A-Za-z0-9+/=\s]+)",
        image,
        re.IGNORECASE
    )

    if match:
        prefix = match.group(1)
        encoded = re.sub(r"\s+", "", match.group(2))

        if not encoded:
            return None

        return prefix + encoded

    # Base64 simples recebido do HTML
    if re.fullmatch(r"[A-Za-z0-9+/=\s]+", image):

        encoded = re.sub(r"\s+", "", image)

        if encoded:
            return f"data:image/jpeg;base64,{encoded}"

    return None


# =========================================================
# COMUNICAÇÃO COM GROQ
# =========================================================

def groq_chat_with_retry(
    messages,
    model,
    temperature=0.4,
    max_tokens=2048,
    retries=3
):

    for attempt in range(retries):

        try:

            response = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:

                retry_after = response.headers.get("Retry-After")

                if retry_after:
                    wait_time = int(retry_after)
                else:
                    wait_time = 2 ** (attempt + 1)

                if attempt < retries - 1:

                    app.logger.warning(
                        f"Rate limit atingido. "
                        f"A aguardar {wait_time} segundos."
                    )

                    time.sleep(wait_time)
                    continue

                raise RuntimeError(
                    "Limite de pedidos da Groq atingido. "
                    "Aguarda um minuto antes de tentar novamente."
                )

            response.raise_for_status()

            data = response.json()

            return data["choices"][0]["message"]["content"].strip()

        except requests.Timeout:

            if attempt < retries - 1:
                time.sleep(2)
                continue

            raise RuntimeError(
                "A IA demorou demasiado a responder."
            )

        except Exception as e:

            if attempt < retries - 1:
                time.sleep(2)
                continue

            raise RuntimeError(
                f"Falha na comunicação com a IA: {str(e)}"
            )


# =========================================================
# HISTÓRICO CONTEXTUAL
# =========================================================

def clean_history(history):

    if not isinstance(history, list):
        return []

    cleaned = []

    # Mantém as últimas 24 mensagens anteriores
    for item in history[-MAX_HISTORY_MESSAGES:]:

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")
        image_src = item.get("imageSrc")

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str):
            continue

        content = content.strip()[:MAX_MESSAGE_LENGTH]

        # Recuperar imagem anterior
        image_url = None

        if role == "user" and image_src:
            image_url = normalizar_imagem(image_src)

        # Mensagem multimodal
        if image_url:

            cleaned.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": content if content else "Imagem enviada anteriormente."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            })

        # Mensagem textual normal
        elif content:

            cleaned.append({
                "role": role,
                "content": content
            })

    return cleaned


# =========================================================
# PESQUISA WEB
# =========================================================

def tavily_search(query):

    if not TAVILY_API_KEY:
        return []

    try:

        response = requests.post(
            TAVILY_URL,
            headers={
                "Content-Type": "application/json"
            },
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 3,
                "include_answer": False
            },
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
            for r in results
            if r.get("url")
        ]

    except Exception:
        return []


# =========================================================
# LEITURA DE SITES
# =========================================================

def ler_site(url):

    try:

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        res = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        res.raise_for_status()

        soup = BeautifulSoup(
            res.text,
            "html.parser"
        )

        for script in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header"
        ]):
            script.decompose()

        return str(
            soup.get_text(
                separator=" ",
                strip=True
            )
        )[:3000]

    except Exception:
        return None


def limpar_url(url):

    return re.sub(
        r"[.,;:!?)}\]>\"'`]+$",
        "",
        str(url).strip()
    )


# =========================================================
# DIAGNÓSTICO TEMPORÁRIO DA GROQ
# =========================================================

@app.route("/diagnostico-groq", methods=["GET", "POST"])
@limiter.limit("3 per hour", methods=["POST"])
def diagnostico_groq():

    if request.method == "GET":

        return """
        <!DOCTYPE html>
        <html lang="pt">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>MGD AI - Diagnóstico Groq</title>
        </head>
        <body style="font-family:Arial;padding:25px;background:#111;color:white">
            <h2>MGD AI - Diagnóstico Groq</h2>
            <p>Introduz o token de diagnóstico configurado no Render.</p>

            <input
                id="token"
                type="password"
                placeholder="Token de diagnóstico"
                style="padding:12px;width:100%;box-sizing:border-box"
            >

            <button
                onclick="executar()"
                style="padding:12px;margin-top:12px"
            >
                Executar diagnóstico
            </button>

            <pre id="resultado" style="white-space:pre-wrap;margin-top:20px"></pre>

            <script>
                async function executar() {
                    const token = document.getElementById("token").value;
                    const resultado = document.getElementById("resultado");

                    resultado.textContent = "A executar diagnóstico...";

                    try {
                        const response = await fetch("/diagnostico-groq", {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json"
                            },
                            body: JSON.stringify({token: token})
                        });

                        const data = await response.json();

                        resultado.textContent =
                            JSON.stringify(data, null, 2);

                    } catch (error) {
                        resultado.textContent =
                            "Falha de ligação: " + error.message;
                    }
                }
            </script>
        </body>
        </html>
        """

    # -----------------------------------------------------
    # VALIDAR TOKEN
    # -----------------------------------------------------

    diagnostic_token = os.getenv(
        "MGD_DIAGNOSTIC_TOKEN",
        ""
    ).strip()

    supplied_token = str(
        (request.get_json(silent=True) or {}).get("token", "")
    ).strip()

    if not diagnostic_token:

        return jsonify({
            "error": "MGD_DIAGNOSTIC_TOKEN nao esta configurado no servidor."
        }), 503

    if not secrets.compare_digest(
        supplied_token,
        diagnostic_token
    ):

        return jsonify({
            "error": "Token de diagnostico invalido."
        }), 403

    # -----------------------------------------------------
    # VALIDAR CHAVE GROQ
    # -----------------------------------------------------

    if not GROQ_API_KEY:

        return jsonify({
            "error": "GROQ_API_KEY nao esta configurada."
        }), 503

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    resultado = {
        "diagnostico": "MGD AI - Groq",
        "modelos_configurados": {
            "texto": GROQ_MODEL_TEXT,
            "visao": GROQ_MODEL_VISION
        },
        "lista_modelos": {},
        "teste_texto": {},
        "teste_visao": {}
    }

    # -----------------------------------------------------
    # TESTAR LISTA DE MODELOS
    # -----------------------------------------------------

    modelos_disponiveis = []

    try:

        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers=headers,
            timeout=20
        )

        if response.status_code == 200:

            modelos_disponiveis = [
                item.get("id")
                for item in response.json().get("data", [])
                if item.get("id")
            ]

            resultado["lista_modelos"] = {
                "status": "sucesso",
                "codigo_http": 200,
                "texto_disponivel": GROQ_MODEL_TEXT in modelos_disponiveis,
                "visao_disponivel": GROQ_MODEL_VISION in modelos_disponiveis
            }

        else:

            resultado["lista_modelos"] = {
                "status": "falhou",
                "codigo_http": response.status_code,
                "detalhes": response.text[:500]
            }

    except Exception as e:

        resultado["lista_modelos"] = {
            "status": "erro",
            "detalhes": str(e)[:300]
        }

    # -----------------------------------------------------
    # FUNÇÃO DE TESTE DE MODELO
    # -----------------------------------------------------

    def testar_modelo(model, messages):

        try:

            response = requests.post(
                GROQ_URL,
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": 64
                },
                timeout=30
            )

            if response.status_code == 200:

                data = response.json()

                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                return {
                    "status": "sucesso",
                    "codigo_http": 200,
                    "resposta": str(content)[:300]
                }

            return {
                "status": "falhou",
                "codigo_http": response.status_code,
                "detalhes": response.text[:500]
            }

        except Exception as e:

            return {
                "status": "erro",
                "detalhes": str(e)[:300]
            }

    # -----------------------------------------------------
    # TESTAR MODELO DE TEXTO
    # -----------------------------------------------------

    resultado["teste_texto"] = testar_modelo(
        GROQ_MODEL_TEXT,
        [
            {
                "role": "user",
                "content": "Responde apenas: TESTE DE TEXTO OK."
            }
        ]
    )

    # -----------------------------------------------------
    # TESTAR MODELO DE VISÃO
    # -----------------------------------------------------

    # PNG válido de 64x64 pixels.
    # Contém um quadrado azul e um círculo amarelo.

    imagem_teste = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAy0lEQVR42u3aMQ6AIBBEUSHe/8pY2FhAiM4gbPjbasg8QVyNqZRyRK58BC8AAAAAmFtn60BKawVtPa5YQgAAAAAAAACAEb2Q3LqkWn9VAgCq0Z+HvIzzn+jjGPnn9J/PHwv4lsZiyLPSuwzbPwf0SyiOsPcMuHYSZRx6IQAAAAQGuDpKZZztl5A+CeII3MTaJdQnME+8Cy17gG0JvU3j2sGcL/V3pm5rue5XiS4jxnehcVnZRgEAAAAAAIDlqtlKRPmdlCUEAAAAAFJdTc87jBJxyJYAAAAASUVORK5CYII="
    )

    resultado["teste_visao"] = testar_modelo(
        GROQ_MODEL_VISION,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Descreve brevemente esta imagem de teste."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": imagem_teste
                        }
                    }
                ]
            }
        ]
    )

    return jsonify(resultado)


# =========================================================
# ROTAS
# =========================================================

@app.get("/")
def home():

    return jsonify({
        "status": "online"
    })


@app.get("/health")
def health():

    return jsonify({
        "status": "ok"
    })


# =========================================================
# CHAT
# =========================================================

@app.post("/chat")
@limiter.limit("20 per minute")
def chat():

    data = request.get_json(silent=True) or {}

    message = str(
        data.get("message", "")
    ).strip()

    image_b64 = data.get("image", None)

    if image_b64 and not isinstance(image_b64, str):
        image_b64 = None

    # Validar imagem atual
    current_image = normalizar_imagem(image_b64)

    # Preparar histórico incluindo imagens anteriores
    history = clean_history(
        data.get("history", [])
    )

    if not message and not current_image:

        return jsonify({
            "error": "Mensagem vazia."
        }), 400

    sources = []
    activities = []

    model_to_use = GROQ_MODEL_TEXT

    final_message = message
    source_context = ""

    try:

        # =================================================
        # ETAPA 1 — DETECTAR IMAGENS NO CONTEXTO
        # =================================================

        history_has_images = any(
            isinstance(item.get("content"), list)
            for item in history
        )

        if current_image:

            activities.extend([
                "A Analisar Imagem...",
                "Analisou a Imagem"
            ])

            model_to_use = GROQ_MODEL_VISION

            final_message = (
                message if message
                else "O que ha nesta imagem?"
            )

        elif history_has_images:

            # Existe uma imagem anterior na conversa.
            # Utilizar o modelo de visão para preservar
            # a capacidade de interpretar esse contexto.

            model_to_use = GROQ_MODEL_VISION

        # =================================================
        # ETAPA 2 — LINKS E PESQUISA
        # =================================================

        if not current_image:

            urls = re.findall(
                r"(https?://[^\s]+)",
                message
            )

            if urls:

                url_alvo = limpar_url(urls[0])

                activities.extend([
                    "A Abrir Link...",
                    "Abriu o Link",
                    "A Extrair conteudo da pagina"
                ])

                conteudo = ler_site(url_alvo)

                if conteudo:

                    activities.extend([
                        "Extraiu o Conteudo",
                        "A Resumir",
                        "Resumiu o Conteudo"
                    ])

                    final_message = (
                        f"O utilizador enviou o link {url_alvo}. "
                        f"Conteudo:\n\n{conteudo}\n\n"
                        f"Responde: {message}."
                    )

                else:

                    activities.append(
                        "Falha ao extrair conteudo"
                    )

                    final_message = (
                        f"O utilizador enviou o link {url_alvo}, "
                        "mas o site bloqueou a leitura. "
                        "Pede para ele copiar e colar o texto. "
                        "PROIBIDO inventar."
                    )

            else:

                precisa_pesquisar = bool(
                    re.search(
                        r"(noticias|hoje|agora|preco|tempo|clima|"
                        r"resultado|jogo|versao|atual|recente|"
                        r"2026|2025)",
                        message,
                        re.IGNORECASE
                    )
                )

                if precisa_pesquisar and TAVILY_API_KEY:

                    activities.extend([
                        "A pesquisar na Web...",
                        "Pesquisou na Web"
                    ])

                    sources = tavily_search(message)

                    if sources:

                        activities.append(
                            f"{len(sources)} Resultados Encontrados"
                        )

                        source_context = "\n\nFONTES:\n"

                        for i, s in enumerate(sources, 1):

                            source_context += (
                                f"[{i}] {s['title']} "
                                f"({s['url']})\n"
                                f"{s['content']}\n\n"
                            )

        # =================================================
        # ETAPA 3 — CONSTRUIR CONTEXTO COMPLETO
        # =================================================

        final_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            *history
        ]

        # Mensagem atual com imagem
        if current_image:

            final_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": final_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": current_image
                        }
                    }
                ]
            })

        # Mensagem atual sem imagem
        else:

            final_messages.append({
                "role": "user",
                "content": final_message + source_context
            })

        # =================================================
        # ETAPA 4 — CHAMAR GROQ
        # =================================================

        reply = groq_chat_with_retry(
            final_messages,
            model_to_use,
            temperature=0.6,
            max_tokens=2048
        )

        reply = str(reply).replace("*", "").replace("#", "")

        if "```" in reply:

            activities.extend([
                "A escrever código...",
                "Escreveu Código",
                "A Verificar código...",
                "Verificou Código"
            ])

        # =================================================
        # ETAPA 5 — RESPOSTA
        # =================================================

        return jsonify({
            "reply": reply,
            "sources": sources,
            "activities": activities
        })

    except Exception as e:

        app.logger.error(
            f"Erro no chat: {e}"
        )

        return jsonify({
            "reply": f"Erro técnico: {str(e)}",
            "sources": [],
            "activities": activities
        }), (
            429
            if "Limite de pedidos" in str(e)
            else 500
        )


# =========================================================
# TEXT TO SPEECH
# =========================================================

@app.post("/tts")
@limiter.limit("20 per minute")
def tts():

    data = request.get_json(silent=True) or {}

    text = str(
        data.get("text", "")
    ).strip()

    if not text:

        return jsonify({
            "error": "Texto vazio."
        }), 400

    if len(text) > 3000:

        text = text[:3000] + "..."

    async def generate_audio():

        communicate = edge_tts.Communicate(
            text,
            TTS_VOICE
        )

        audio_data = b""

        async for chunk in communicate.stream():

            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        return audio_data

    try:

        loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)

        audio_bytes = loop.run_until_complete(
            generate_audio()
        )

        loop.close()

        return send_file(
            io.BytesIO(audio_bytes),
            mimetype="audio/mpeg",
            as_attachment=False
        )

    except Exception as e:

        app.logger.error(
            f"Erro TTS: {e}"
        )

        return jsonify({
            "error": "Erro ao gerar voz."
        }), 500


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000"))
    )
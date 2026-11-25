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
from flask import Flask, Response, jsonify, request, send_file, send_from_directory, stream_with_context
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

MAX_HISTORY_MESSAGES = 10
MAX_MESSAGE_LENGTH = 4000
MAX_HISTORY_CHARS = 12000

# Limite para imagens recebidas no histórico

MAX_IMAGE_LENGTH = 2_000_000


SYSTEM_PROMPT = """
Es a MGD AI, a inteligencia artificial oficial da MGD REDLINE.
O teu criador e Wayne Matimbe, tambem conhecido como CEO Wayne.

IDENTIDADE E AUTORREFERENCIA:
- Fala como a própria MGD AI, não como um observador externo.
- Quando falares de ti, utiliza naturalmente a primeira pessoa:
  "eu", "meu", "minha", "comigo", "posso" e "consigo".
- Não te descrevas na terceira pessoa. Evita frases como
  "a MGD AI consegue", "a MGD AI foi criada" ou "ela pode ajudar"
  quando estiveres a falar de ti própria.
- Prefere "Eu consigo ajudar", "Fui criada", "As minhas capacidades"
  e "Posso pesquisar na internet".
- Quando o utilizador perguntar sobre a tua origem, criação,
  capacidades, identidade ou funcionamento, responde diretamente
  como a própria MGD AI, em primeira pessoa.
- Distingue sempre a tua identidade da identidade do teu criador.
  Wayne Matimbe, também conhecido como CEO Wayne, é o teu criador;
  não fales dele como se fosses tu.
- Mantém esta identidade em qualquer idioma. Adapta a gramática
  e os pronomes à língua da conversa, sem perder a primeira pessoa.
- Não forces a repetição de "eu" em todas as frases: escreve de forma
  natural, mas nunca te apresentes como uma entidade externa.

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

PESQUISA E ATUALIDADE:
- A pesquisa na Web e uma ferramenta que deves solicitar quando a resposta
  depender de factos externos atuais, verificacao de fontes, acontecimentos
  recentes, disponibilidade, precos, horarios, versoes ou dados que possam mudar.
- Nao solicites pesquisa apenas porque uma palavra especifica apareceu.
- Tambem nao evites pesquisar apenas porque o utilizador formulou a pergunta
  de maneira indireta ou continuou um assunto anterior. Interpreta a intencao
  e o contexto completo da conversa.
- Se nao houver pesquisa ou fontes verificaveis, nao afirmes que verificaste
  informacoes em tempo real. Distingue conhecimento geral de factos atuais.

MEMORIA VISUAL TEXTUAL:
- Quando receberes uma imagem nova, analisa-a visualmente apenas nesse pedido.
- Responde ao pedido do utilizador e termina com uma secção "Registo visual".
- Nesse registo, descreve os elementos realmente visíveis e relevantes,
  incluindo texto legível, posições, cores, objetos, detalhes e relações espaciais.
- Separa claramente "Identificado", "Incerto" e "Não verificado".
- Não inventes detalhes nem declares como observado algo que não conseguiste ver.
- O registo visual será guardado como texto no histórico e será a única memória
  disponível para perguntas posteriores sobre essa imagem.
- Em mensagens posteriores, usa apenas o registo textual e o histórico recebido.
  Não assumes que ainda tens acesso à imagem original.
- Se a resposta exigir um detalhe que não consta do registo ou que ficou incerto,
  explica que não o consegues confirmar sem a imagem e pede ao utilizador
  que a envie novamente. Nunca inventes uma verificação visual.
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

            if response.status_code == 413:
                # 413 não melhora com repetição idêntica. Reduz o histórico,
                # preservando o system prompt e a mensagem actual (incluindo imagem).
                if not locals().get("compact_retry_done", False):
                    compact_retry_done = True
                    system_msg = next((m for m in messages if m.get("role") == "system"), None)
                    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
                    compact_messages = [m for m in (system_msg, last_user) if m]
                    messages = compact_messages
                    max_tokens = min(max_tokens, 1024)
                    app.logger.warning("Groq 413: a repetir uma vez sem histórico antigo e com saída reduzida.")
                    continue
                raise RuntimeError(
                    "O pedido continua demasiado grande para a Groq, mesmo após reduzir o histórico. "
                    "Envia uma imagem mais pequena ou inicia uma nova conversa."
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

            if "413" in str(e) or "demasiado grande para a Groq" in str(e):
                raise RuntimeError(str(e))

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
    """Mantém apenas contexto recente e limita o tamanho total enviado à Groq."""
    if not isinstance(history, list):
        return []

    cleaned = []
    total_chars = 0

    # Só texto recente. Imagens antigas não são reenviadas em cada pedido:
    # isso fazia o payload crescer continuamente ao longo da conversa.
    for item in history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue

        content = content.strip()[:MAX_MESSAGE_LENGTH]
        if not content:
            continue

        remaining = MAX_HISTORY_CHARS - total_chars
        if remaining <= 0:
            break
        content = content[:remaining]
        total_chars += len(content)
        cleaned.append({"role": role, "content": content})

    return cleaned


# =========================================================
# DECISAO CONTEXTUAL DE PESQUISA
# =========================================================

RESEARCH_PLANNER_PROMPT = """
Es um modulo interno de decisao de ferramentas da MGD AI. Analisa a mensagem
mais recente JUNTO COM o historico fornecido. Nao respondas ao utilizador.
Decide se a tarefa exige pesquisa na Web para produzir uma resposta correta.

Usa pesquisa quando forem necessarios factos externos verificaveis que possam
ter mudado, confirmacao de informacao atual, acontecimentos, disponibilidade,
precos, horarios, versoes, resultados, politicas atuais, comparacoes de ofertas
ou quando o utilizador pedir explicitamente para pesquisar/verificar fontes.
Nao uses pesquisa para explicacoes estaveis, calculos, escrita, traducao,
conversa, opinioes claramente subjetivas ou conhecimento geral suficiente.
Uma pergunta de seguimento pode exigir pesquisa mesmo sem repetir o assunto.
Nao decidas por palavras isoladas: interpreta a intencao e o contexto.

Devolve SOMENTE um objeto JSON neste formato:
{"search": true, "query": "consulta curta e especifica"}
ou
{"search": false, "query": ""}
Nao inventes a necessidade de pesquisa. Se houver duvida real sobre atualidade
ou verificacao externa, prefere pesquisar. A consulta deve refletir o pedido,
nao acrescentar factos presumidos.
"""


def decidir_pesquisa_contextual(message, history):
    """Pede ao modelo que decida com base na intencao e no contexto, nao em regex."""
    planner_messages = [
        {"role": "system", "content": RESEARCH_PLANNER_PROMPT},
        *history[-8:],
        {"role": "user", "content": message[:MAX_MESSAGE_LENGTH]}
    ]

    raw = groq_chat_with_retry(
        planner_messages,
        GROQ_MODEL_TEXT,
        temperature=0,
        max_tokens=120,
        retries=2
    )

    # Aceita JSON puro ou JSON envolvido acidentalmente em bloco de codigo.
    candidate = str(raw).strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    match = re.search(r"\{.*?\}", candidate, re.DOTALL)
    if not match:
        raise ValueError("A decisao contextual de pesquisa devolveu um formato invalido.")

    decision = json.loads(match.group(0))
    should_search = decision.get("search") is True
    query = str(decision.get("query") or "").strip()
    if should_search and not query:
        query = message[:500]
    return should_search, query[:500]


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
# MGD AI COMMUNITY EDITION - MOTOR JSON
# Base inicial: membros, quiz, ranking e registo de moderacao.
# A ligacao WhatsApp sera adicionada separadamente.
# =========================================================

import threading
import uuid
from datetime import datetime, timezone, timedelta

COMMUNITY_DATA_DIR = os.getenv(
    "COMMUNITY_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance", "community_data")
)
COMMUNITY_API_TOKEN = os.getenv("COMMUNITY_API_TOKEN", "").strip()
COMMUNITY_LOCK = threading.RLock()

COMMUNITY_DEFAULTS = {
    "community_members.json": {"members": {}},
    "anime_quiz.json": {
        "questions": [
            {
                "id": "anime001",
                "question": "Qual e o nome do protagonista de One Piece?",
                "options": {"A": "Roronoa Zoro", "B": "Monkey D. Luffy", "C": "Sanji", "D": "Trafalgar Law"},
                "answer": "B",
                "explanation": "Monkey D. Luffy e o protagonista de One Piece.",
                "difficulty": "facil",
                "category": "One Piece"
            },
            {
                "id": "anime002",
                "question": "Como se chama a aldeia natal de Naruto?",
                "options": {"A": "Sunagakure", "B": "Kirigakure", "C": "Konohagakure", "D": "Iwagakure"},
                "answer": "C",
                "explanation": "Naruto cresceu em Konohagakure, a Vila Oculta da Folha.",
                "difficulty": "facil",
                "category": "Naruto"
            },
            {
                "id": "anime003",
                "question": "Quem criou Dragon Ball?",
                "options": {"A": "Eiichiro Oda", "B": "Masashi Kishimoto", "C": "Tite Kubo", "D": "Akira Toriyama"},
                "answer": "D",
                "explanation": "Dragon Ball foi criado por Akira Toriyama.",
                "difficulty": "facil",
                "category": "Dragon Ball"
            },
            {
                "id": "anime004",
                "question": "Em Death Note, o que permite ao utilizador do caderno matar uma pessoa?",
                "options": {"A": "Escrever o nome da pessoa, tendo o rosto em mente", "B": "Desenhar o rosto", "C": "Dizer o nome em voz alta", "D": "Tocar numa fotografia"},
                "answer": "A",
                "explanation": "A regra central do Death Note exige escrever o nome enquanto se visualiza o rosto da pessoa.",
                "difficulty": "medio",
                "category": "Death Note"
            },
            {
                "id": "anime005",
                "question": "Qual e o nome do protagonista de Bleach?",
                "options": {"A": "Uryu Ishida", "B": "Renji Abarai", "C": "Ichigo Kurosaki", "D": "Byakuya Kuchiki"},
                "answer": "C",
                "explanation": "Ichigo Kurosaki e o protagonista de Bleach.",
                "difficulty": "facil",
                "category": "Bleach"
            }
        ],
        "active_quiz": None,
        "history": [],
        "last_auto_quiz_at": None
    },
    "community_ranking.json": {"scores": {}},
    "community_moderation.json": {"events": []},
    "community_config.json": {
        "community_name": "MGD AI Community Edition",
        "quiz_auto_enabled": True,
        "quiz_interval_minutes": 60,
        "points_per_correct_answer": 10,
        "moderation_mode": "hybrid",
        "whatsapp_integration": "undecided"
    }
}


def community_json_path(filename):
    return os.path.join(COMMUNITY_DATA_DIR, filename)


def community_load(filename):
    """Le JSON; cria o ficheiro com os valores iniciais quando ainda nao existe."""
    if filename not in COMMUNITY_DEFAULTS:
        raise ValueError("Ficheiro Community Edition desconhecido.")
    os.makedirs(COMMUNITY_DATA_DIR, exist_ok=True)
    path = community_json_path(filename)
    with COMMUNITY_LOCK:
        if not os.path.exists(path):
            community_save(filename, COMMUNITY_DEFAULTS[filename])
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else json.loads(json.dumps(COMMUNITY_DEFAULTS[filename]))
        except (OSError, json.JSONDecodeError):
            app.logger.exception("Falha ao ler %s; a usar valores iniciais.", filename)
            return json.loads(json.dumps(COMMUNITY_DEFAULTS[filename]))


def community_save(filename, data):
    """Gravacao atomica para reduzir o risco de JSON incompleto."""
    if filename not in COMMUNITY_DEFAULTS:
        raise ValueError("Ficheiro Community Edition desconhecido.")
    os.makedirs(COMMUNITY_DATA_DIR, exist_ok=True)
    path = community_json_path(filename)
    temp_path = path + ".tmp"
    with COMMUNITY_LOCK:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)


def community_require_token():
    """Protege rotas de escrita. Exige COMMUNITY_API_TOKEN no servidor."""
    if not COMMUNITY_API_TOKEN:
        return jsonify({"error": "Configura COMMUNITY_API_TOKEN nas variaveis de ambiente."}), 503
    supplied = request.headers.get("Authorization", "")
    expected = "Bearer " + COMMUNITY_API_TOKEN
    if not secrets.compare_digest(supplied, expected):
        return jsonify({"error": "Acesso nao autorizado."}), 401
    return None


def community_normalize_answer(value):
    value = str(value or "").strip().casefold()
    value = re.sub(r"\s+", " ", value)
    return value


def community_register_member(member_id, display_name):
    member_id = str(member_id or "").strip()[:160]
    display_name = str(display_name or "Membro") .strip()[:80]
    if not member_id:
        raise ValueError("member_id e obrigatorio.")
    with COMMUNITY_LOCK:
        data = community_load("community_members.json")
        members = data.setdefault("members", {})
        member = members.get(member_id)
        if not isinstance(member, dict):
            member = {
                "member_id": member_id,
                "display_name": display_name or "Membro",
                "joined_at": datetime.now(timezone.utc).isoformat(),
                "messages_seen": 0,
                "quiz_correct": 0,
                "quiz_answered": 0
            }
        else:
            member["display_name"] = display_name or member.get("display_name", "Membro")
        members[member_id] = member
        community_save("community_members.json", data)
        return member


def community_create_quiz(is_automatic=False):
    with COMMUNITY_LOCK:
        quiz_data = community_load("anime_quiz.json")
        questions = quiz_data.get("questions", [])
        if not questions:
            raise ValueError("Ainda nao existem perguntas no banco de quizzes.")
        previous_ids = [item.get("question_id") for item in quiz_data.get("history", [])[-5:]]
        candidates = [item for item in questions if item.get("id") not in previous_ids]
        if not candidates:
            candidates = questions
        question = __import__("random").choice(candidates)
        quiz_id = str(uuid.uuid4())
        active = {
            "quiz_id": quiz_id,
            "question_id": question.get("id"),
            "question": question.get("question", ""),
            "options": question.get("options", {}),
            "answer": question.get("answer", ""),
            "explanation": question.get("explanation", ""),
            "category": question.get("category", "Anime"),
            "difficulty": question.get("difficulty", "normal"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "automatic": bool(is_automatic),
            "responses": {}
        }
        quiz_data["active_quiz"] = active
        quiz_data.setdefault("history", []).append({
            "quiz_id": quiz_id,
            "question_id": question.get("id"),
            "created_at": active["created_at"],
            "automatic": bool(is_automatic)
        })
        quiz_data["history"] = quiz_data["history"][-200:]
        community_save("anime_quiz.json", quiz_data)
        return {
            "quiz_id": quiz_id,
            "question": active["question"],
            "options": active["options"],
            "category": active["category"],
            "difficulty": active["difficulty"],
            "automatic": active["automatic"]
        }


def community_submit_answer(member_id, display_name, quiz_id, answer):
    member = community_register_member(member_id, display_name)
    with COMMUNITY_LOCK:
        quiz_data = community_load("anime_quiz.json")
        active = quiz_data.get("active_quiz")
        if not active or active.get("quiz_id") != str(quiz_id):
            raise ValueError("Este quiz ja nao esta ativo. Pede !quiz para iniciar outro.")
        responses = active.setdefault("responses", {})
        if member_id in responses:
            raise ValueError("Ja respondeste a este quiz.")
        submitted = community_normalize_answer(answer)
        correct_key = community_normalize_answer(active.get("answer"))
        options = active.get("options", {})
        correct_text = community_normalize_answer(options.get(active.get("answer"), ""))
        is_correct = submitted == correct_key or submitted == correct_text
        if submitted in {community_normalize_answer(k) for k in options}:
            is_correct = submitted == correct_key
        responses[member_id] = {
            "answer": str(answer)[:200],
            "correct": bool(is_correct),
            "answered_at": datetime.now(timezone.utc).isoformat()
        }
        community_save("anime_quiz.json", quiz_data)

        members_data = community_load("community_members.json")
        stored_member = members_data.setdefault("members", {}).setdefault(member_id, member)
        stored_member["quiz_answered"] = int(stored_member.get("quiz_answered", 0)) + 1
        if is_correct:
            stored_member["quiz_correct"] = int(stored_member.get("quiz_correct", 0)) + 1
        community_save("community_members.json", members_data)

        ranking = community_load("community_ranking.json")
        scores = ranking.setdefault("scores", {})
        score = scores.setdefault(member_id, {"display_name": member.get("display_name", "Membro"), "points": 0, "correct_answers": 0})
        score["display_name"] = member.get("display_name", "Membro")
        if is_correct:
            config = community_load("community_config.json")
            points = max(0, int(config.get("points_per_correct_answer", 10)))
            score["points"] = int(score.get("points", 0)) + points
            score["correct_answers"] = int(score.get("correct_answers", 0)) + 1
        community_save("community_ranking.json", ranking)
        return {
            "correct": bool(is_correct),
            "correct_answer": active.get("answer"),
            "correct_text": options.get(active.get("answer"), ""),
            "explanation": active.get("explanation", ""),
            "points_awarded": points if is_correct else 0,
            "total_points": int(score.get("points", 0))
        }


def community_leaderboard(limit=10):
    ranking = community_load("community_ranking.json").get("scores", {})
    rows = [
        {"member_id": key, "display_name": value.get("display_name", "Membro"),
         "points": int(value.get("points", 0)),
         "correct_answers": int(value.get("correct_answers", 0))}
        for key, value in ranking.items() if isinstance(value, dict)
    ]
    rows.sort(key=lambda item: (item["points"], item["correct_answers"]), reverse=True)
    return rows[:max(1, min(int(limit), 50))]


def community_command_result(text, member_id, display_name):
    """Interpreta comandos do grupo sem depender de palavras-chave soltas."""
    raw = str(text or "").strip()
    command, _, argument = raw.partition(" ")
    command = command.casefold()
    if command == "!ajuda":
        return {"type": "text", "reply": "Comandos disponiveis: !quiz, !responder A, !ranking, !perfil e !ajuda."}
    if command == "!quiz":
        quiz = community_create_quiz(False)
        lines = [quiz["question"]]
        lines.extend([f"{key}) {value}" for key, value in quiz["options"].items()])
        lines.append("Responde com !responder A, B, C ou D.")
        return {"type": "quiz", "quiz": quiz, "reply": "\n".join(lines)}
    if command == "!ranking":
        rows = community_leaderboard(10)
        if not rows:
            return {"type": "text", "reply": "O ranking ainda esta vazio. Participa no primeiro quiz!"}
        lines = ["RANKING MGD AI COMMUNITY"]
        lines.extend([f"{i}. {row['display_name']} - {row['points']} pontos" for i, row in enumerate(rows, 1)])
        return {"type": "ranking", "ranking": rows, "reply": "\n".join(lines)}
    if command == "!perfil":
        member = community_register_member(member_id, display_name)
        score = community_load("community_ranking.json").get("scores", {}).get(member_id, {})
        return {"type": "profile", "reply": f"{member.get('display_name')}: {score.get('points', 0)} pontos, {member.get('quiz_correct', 0)} respostas certas."}
    if command == "!responder":
        quiz = community_load("anime_quiz.json").get("active_quiz") or {}
        if not argument.strip():
            raise ValueError("Usa !responder A, B, C ou D.")
        result = community_submit_answer(member_id, display_name, quiz.get("quiz_id"), argument.strip())
        if result["correct"]:
            reply = f"Resposta certa! +{result['points_awarded']} pontos. {result['explanation']}"
        else:
            reply = f"Ainda nao foi desta! A resposta correta era {result['correct_answer']}) {result['correct_text']}. {result['explanation']}"
        return {"type": "answer", "result": result, "reply": reply}
    return {"type": "unknown", "reply": "Nao reconheci esse comando. Escreve !ajuda para ver os comandos."}


@app.get("/community/health")
def community_health():
    members = community_load("community_members.json").get("members", {})
    quiz = community_load("anime_quiz.json")
    return jsonify({
        "status": "ok",
        "community_engine": "ready",
        "members": len(members),
        "quiz_questions": len(quiz.get("questions", [])),
        "active_quiz": bool(quiz.get("active_quiz")),
        "storage": "json",
        "whatsapp": "not_connected"
    })


@app.get("/community/ranking")
def community_ranking_route():
    return jsonify({"ranking": community_leaderboard(request.args.get("limit", 10))})


@app.post("/community/member/register")
@limiter.limit("30 per minute")
def community_member_register_route():
    denied = community_require_token()
    if denied:
        return denied
    data = request.get_json(silent=True) or {}
    try:
        member = community_register_member(data.get("member_id"), data.get("display_name"))
        return jsonify({"ok": True, "member": member})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.post("/community/quiz/start")
@limiter.limit("10 per minute")
def community_quiz_start_route():
    denied = community_require_token()
    if denied:
        return denied
    try:
        return jsonify({"ok": True, "quiz": community_create_quiz(False)})
    except ValueError as error:
        return jsonify({"error": str(error)}), 503


@app.post("/community/quiz/answer")
@limiter.limit("30 per minute")
def community_quiz_answer_route():
    denied = community_require_token()
    if denied:
        return denied
    data = request.get_json(silent=True) or {}
    try:
        result = community_submit_answer(
            str(data.get("member_id", "")).strip(),
            data.get("display_name", "Membro"),
            data.get("quiz_id"),
            data.get("answer")
        )
        return jsonify({"ok": True, "result": result})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.post("/community/command")
@limiter.limit("40 per minute")
def community_command_route():
    denied = community_require_token()
    if denied:
        return denied
    data = request.get_json(silent=True) or {}
    try:
        result = community_command_result(
            data.get("text", ""),
            str(data.get("member_id", "")).strip(),
            data.get("display_name", "Membro")
        )
        return jsonify({"ok": True, **result})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.post("/community/quiz/auto")
@limiter.limit("5 per minute")
def community_quiz_auto_route():
    """A futura WhatsApp Bridge consulta esta rota no intervalo configurado."""
    denied = community_require_token()
    if denied:
        return denied
    config = community_load("community_config.json")
    if not config.get("quiz_auto_enabled", True):
        return jsonify({"ok": True, "due": False, "reason": "automatic_quiz_disabled"})
    quiz_data = community_load("anime_quiz.json")
    last = quiz_data.get("last_auto_quiz_at")
    interval = max(5, int(config.get("quiz_interval_minutes", 60)))
    if last:
        try:
            last_dt = datetime.fromisoformat(last)
            if datetime.now(timezone.utc) - last_dt < timedelta(minutes=interval):
                return jsonify({"ok": True, "due": False, "reason": "interval_not_reached"})
        except (TypeError, ValueError):
            pass
    try:
        quiz = community_create_quiz(True)
        quiz_data = community_load("anime_quiz.json")
        quiz_data["last_auto_quiz_at"] = datetime.now(timezone.utc).isoformat()
        community_save("anime_quiz.json", quiz_data)
        return jsonify({"ok": True, "due": True, "quiz": quiz})
    except ValueError as error:
        return jsonify({"error": str(error)}), 503


@app.post("/community/moderation/report")
@limiter.limit("30 per minute")
def community_moderation_report_route():
    """Regista ocorrencias para revisao humana; nao expulsa nem pune membros."""
    denied = community_require_token()
    if denied:
        return denied
    data = request.get_json(silent=True) or {}
    event = {
        "event_id": str(uuid.uuid4()),
        "member_id": str(data.get("member_id", ""))[:160],
        "message_id": str(data.get("message_id", ""))[:160],
        "category": str(data.get("category", "needs_review"))[:80],
        "description": str(data.get("description", ""))[:1000],
        "status": "pending_review",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    moderation = community_load("community_moderation.json")
    moderation.setdefault("events", []).append(event)
    moderation["events"] = moderation["events"][-1000:]
    community_save("community_moderation.json", moderation)
    return jsonify({"ok": True, "event": event})


# =========================================================
# FIM DO MOTOR MGD AI COMMUNITY EDITION
# =========================================================


# =========================================================
# ROTAS
# =========================================================

@app.get("/")
def home():
    # Serve a interface web a partir do mesmo serviço Render.
    # O index.html deve estar na mesma pasta deste app.py.
    return send_from_directory(
        os.path.dirname(os.path.abspath(__file__)),
        "index.html"
    )


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

    message = str(data.get("message", "")).strip()
    image_b64 = data.get("image")

    if image_b64 and not isinstance(image_b64, str):
        image_b64 = None

    current_image = normalizar_imagem(image_b64)
    history = clean_history(data.get("history", []))

    if not message and not current_image:
        return jsonify({"error": "Mensagem vazia."}), 400

    def event_stream():
        """Envia eventos apenas quando cada etapa começa ou termina."""
        sources = []
        model_to_use = GROQ_MODEL_TEXT
        final_message = message
        source_context = ""
        activities = []

        def emit(event_name, payload):
            return (
                f"event: {event_name}\n"
                f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            )

        def activity(label):
            activities.append(label)
            return emit("activity", {"message": label})

        try:
            yield activity("Pedido recebido; a interpretar o contexto")

            if current_image:
                model_to_use = GROQ_MODEL_VISION
                final_message = message or "O que há nesta imagem?"
                final_message += (
                    "\n\nINSTRUÇÃO DE MEMÓRIA VISUAL: analisa a imagem agora. "
                    "Responde primeiro ao pedido e termina com a secção 'Registo visual', "
                    "dividida em 'Identificado', 'Incerto' e 'Não verificado'. "
                    "Regista detalhes concretos e úteis que realmente consigas observar. "
                    "Não adivinhes texto pequeno, números, identidades ou detalhes ocultos."
                )
                yield activity("Imagem recebida e validada; análise visual preparada")

            if not current_image:
                urls = re.findall(r"(https?://[^\s]+)", message)

                if urls:
                    url_alvo = limpar_url(urls[0])
                    yield activity("A abrir o link enviado")
                    conteudo = ler_site(url_alvo)

                    if conteudo:
                        yield activity("Página lida; conteúdo extraído")
                        final_message = (
                            f"O utilizador enviou o link {url_alvo}. "
                            f"Conteúdo:\n\n{conteudo}\n\nResponde: {message}."
                        )
                    else:
                        yield activity("Não foi possível extrair o conteúdo do site")
                        final_message = (
                            f"O utilizador enviou o link {url_alvo}, mas o site bloqueou a leitura. "
                            "Pede para ele copiar e colar o texto. PROIBIDO inventar."
                        )
                else:
                    # A necessidade de pesquisa e decidida pelo modelo com base
                    # na intencao e no historico, nunca por uma lista de keywords.
                    yield activity("A avaliar se a resposta exige verificacao externa")
                    try:
                        precisa_pesquisar, consulta_web = decidir_pesquisa_contextual(
                            message, history
                        )
                    except Exception as planner_error:
                        app.logger.warning(
                            "Nao foi possivel concluir a decisao contextual de pesquisa: %s",
                            planner_error
                        )
                        precisa_pesquisar, consulta_web = False, ""
                        yield activity("Decisao de pesquisa indisponivel; a responder sem alegar verificacao Web")

                    if precisa_pesquisar and TAVILY_API_KEY:
                        yield activity("A pesquisar na Web com base no contexto")
                        sources = tavily_search(consulta_web)
                        if sources:
                            yield activity(f"Pesquisa concluída; {len(sources)} fontes encontradas")
                            source_context = "\n\nFONTES VERIFICADAS PARA ESTA RESPOSTA:\n"
                            for i, source in enumerate(sources, 1):
                                source_context += (
                                    f"[{i}] {source['title']} ({source['url']})\n"
                                    f"{source['content']}\n\n"
                                )
                        else:
                            yield activity("A pesquisa foi solicitada, mas não encontrou resultados utilizáveis")
                    elif precisa_pesquisar and not TAVILY_API_KEY:
                        yield activity("A pesquisa seria necessária, mas a ferramenta Web não está configurada")

            yield activity("A preparar o contexto para a resposta")

            final_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *history
            ]

            if current_image:
                final_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_message},
                        {"type": "image_url", "image_url": {"url": current_image}}
                    ]
                })
            else:
                final_messages.append({
                    "role": "user",
                    "content": final_message + source_context
                })

            yield activity("A gerar a resposta com a IA")
            reply = groq_chat_with_retry(
                final_messages,
                model_to_use,
                temperature=0.6,
                max_tokens=2048
            )
            reply = str(reply).replace("*", "").replace("#", "")

            yield activity("Resposta gerada com sucesso")
            yield emit("done", {
                "reply": reply,
                "sources": sources,
                "activities": activities
            })

        except Exception as e:
            app.logger.error(f"Erro no chat: {e}")
            yield emit("error", {
                "message": (
                    "Limite de pedidos atingido. Aguarda um minuto."
                    if "Limite de pedidos" in str(e)
                    else f"Erro técnico: {str(e)}"
                )
            })

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
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
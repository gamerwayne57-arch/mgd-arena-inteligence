import os
import json
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "*").split(","))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

GROQ_MODEL = "openai/gpt-oss-120b"  # Modelo oficial escolhido para a MGD AI ARENA
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TAVILY_URL = "https://api.tavily.com/search"

REQUEST_TIMEOUT = 35
MAX_HISTORY_MESSAGES = 12
MAX_MESSAGE_LENGTH = 6000

SYSTEM_PROMPT = """
És a MGD AI da MGD ARENA, a inteligência artificial gamer da MGD REDLINE,
criada por Wayne Matimbe, também conhecido como CEO Wayne.

MISSÃO E IDENTIDADE
- És especialista em jogos e em tudo o que acontece dentro da MGD ARENA:
  jogos, jogadores, rankings, XP, medalhas, torneios, regras, funcionalidades,
  estratégias e comunidade.
- Não és uma assistente geral. Se te pedirem assuntos sem relação com jogos
  ou com a MGD ARENA, explica com simpatia que o teu foco é o universo gamer.
- Responde sempre no mesmo idioma da mensagem mais recente do utilizador.
  Se o utilizador mudar de idioma, acompanha-o automaticamente. Isto inclui
  português, inglês, francês, espanhol e qualquer outra língua que consigas
  compreender. Não assumes português como idioma padrão quando a mensagem
  estiver noutro idioma. Se a mensagem misturar idiomas, usa o idioma dominante.
- Mantém o idioma do utilizador mesmo quando consultares fontes noutras línguas.
- Nunca uses asteriscos (*) para destacar texto. Não uses negrito ou itálico
  Markdown. Prefere texto simples, títulos simples e listas sem asteriscos.
- Slogan da Arena: “Aqui não é só jogar... É FAZER HISTÓRIA! 🔥”
  Usa-o raramente, apenas quando fizer sentido.

PRECISÃO E HONESTIDADE
- Nunca inventes jogadores, classificações, resultados, torneios, regras,
  funcionalidades, links, preços, datas, versões, tamanhos ou requisitos.
- Distingue claramente factos confirmados de informação incerta.
- Se não souberes ou não conseguires confirmar algo, diz isso diretamente.
- Não afirmes que pesquisaste na internet se não recebeste resultados de pesquisa.
- Para downloads e atualizações, dá prioridade a fontes oficiais e seguras.
- Ao falar do espaço de um jogo, distingue tamanho indicado na loja,
  downloads adicionais e espaço total recomendado no dispositivo.
- Não assumes que uma lista fixa de jogos representa tudo o que está disponível.
- Não afirmes que uma funcionalidade da MGD ARENA existe se não estiver
  confirmada no contexto ou nos dados fornecidos.

REGRAS DE FORMATAÇÃO
- Nunca uses o caractere asterisco (*) para destacar palavras ou frases.
- Não uses Markdown com asteriscos, nem negrito ou itálico.
- Escreve em texto simples, com títulos curtos quando necessário.

CATÁLOGO DA MGD ARENA — REGRA CRÍTICA
- Não confundas jogos populares, jogos encontrados na internet ou jogos de outras arenas com jogos disponíveis na MGD ARENA.
- Só podes afirmar que um jogo está disponível NA MGD ARENA se essa disponibilidade estiver confirmada por dados oficiais da própria MGD ARENA fornecidos no contexto ou por uma fonte oficial do próprio site.
- Resultados do Tavily sobre outras arenas, plataformas, lojas ou comunidades nunca comprovam que um jogo está disponível na MGD ARENA.
- Se não tiveres acesso ao catálogo oficial da MGD ARENA, diz claramente que ainda não consegues confirmar a lista de jogos disponíveis. Não inventes nem apresentes sugestões externas como se fossem jogos da Arena.
- Podes falar de jogos conhecidos em geral quando a pergunta for sobre jogos em geral, mas distingue isso claramente do catálogo da MGD ARENA.

Responde sempre de forma útil, direta e focada no universo gamer.
""".strip()

DECISION_PROMPT = """
Decide se a pergunta precisa de pesquisa atual na web para ser respondida
com segurança. Pesquisa quando envolver informação que muda com o tempo
(ex.: disponibilidade atual, links oficiais, requisitos/versões recentes,
preços, notícias, atualizações, estado de serviços ou factos atuais sobre jogos).
Não pesquises para conversa casual, explicações gerais estáveis, nem para
informação que já esteja claramente no contexto.
Perguntas sobre quais jogos, funcionalidades, torneios ou opções estão disponíveis
DENTRO DA MGD ARENA exigem dados oficiais internos da própria Arena. Uma pesquisa
web genérica não consegue confirmar o catálogo interno; não pesquises na web para
tentar adivinhar essa disponibilidade.
Não uses palavras-chave fixas como única regra: interpreta a intenção completa.
Devolve APENAS JSON válido neste formato:
{"search": true, "query": "consulta objetiva para pesquisar"}
ou
{"search": false, "query": ""}
""".strip()


def groq_chat(messages, temperature=0.4, max_tokens=1200):
    if not GROQ_API_KEY:
        raise RuntimeError("A chave GROQ_API_KEY não está configurada.")
    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
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
    raw = groq_chat(messages, temperature=0, max_tokens=150)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        decision = json.loads(raw[start:end])
        should_search = decision.get("search") is True
        query = decision.get("query", "")
        if not isinstance(query, str):
            query = ""
        return should_search, query.strip()
    except (ValueError, TypeError, json.JSONDecodeError):
        # Em caso de decisão malformada, não pesquisar por suposição.
        return False, ""


def tavily_search(query):
    if not TAVILY_API_KEY:
        return []
    response = requests.post(
        TAVILY_URL,
        headers={"Content-Type": "application/json"},
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
            "max_results": 5,
            "include_answer": False,
        },
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Erro Tavily ({response.status_code}): {response.text[:500]}")
    results = response.json().get("results", [])
    sources = []
    for result in results[:5]:
        url = result.get("url", "")
        title = result.get("title", "Fonte")
        content = result.get("content", "")
        if url:
            sources.append({
                "title": str(title)[:250],
                "url": str(url),
                "content": str(content)[:1200],
            })
    return sources


@app.get("/")
def home():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.isfile(index_path):
        return send_from_directory(BASE_DIR, "index.html")
    return jsonify({
        "service": "MGD ARENA INTELLIGENCE",
        "status": "online",
        "message": "Backend ativo. Coloca o ficheiro index.html na mesma pasta que app.py para abrir a interface.",
        "health": "/health",
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "groq_configured": bool(GROQ_API_KEY),
        "web_search_configured": bool(TAVILY_API_KEY),
    })


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Escreve uma mensagem para começar."}), 400

    message = message.strip()[:MAX_MESSAGE_LENGTH]
    history = clean_history(data.get("history", []))

    if not GROQ_API_KEY:
        return jsonify({
            "reply": "A ligação à minha inteligência ainda não está configurada. Tenta novamente mais tarde.",
            "sources": [],
            "searched": False,
        }), 503

    sources = []
    searched = False
    search_note = ""

    try:
        should_search, search_query = decide_search(history, message)

        if should_search:
            if TAVILY_API_KEY and search_query:
                try:
                    sources = tavily_search(search_query)
                    searched = True
                    if not sources:
                        search_note = (
                            "A pesquisa foi tentada, mas não devolveu fontes utilizáveis. "
                            "Não inventes dados nem links; informa o utilizador da limitação."
                        )
                except Exception as exc:
                    app.logger.warning("Falha na pesquisa Tavily: %s", exc)
                    search_note = (
                        "A pesquisa web falhou nesta tentativa. Não afirmes que confirmaste "
                        "informação atual e não inventes links ou dados."
                    )
            else:
                search_note = (
                    "A pergunta exige informação atual, mas a pesquisa web não está disponível. "
                    "Sê transparente e não inventes uma resposta."
                )

        source_context = ""
        if sources:
            source_context = "\n\nFONTES ENCONTRADAS (usa apenas o que estas fontes sustentam):\n"
            for i, source in enumerate(sources, 1):
                source_context += (
                    f"{i}. {source['title']}\nURL: {source['url']}\n"
                    f"Conteúdo: {source['content']}\n\n"
                )

        final_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {
                "role": "user",
                "content": (
                    message
                    + (f"\n\nNOTA DE PESQUISA: {search_note}" if search_note else "")
                    + source_context
                ),
            },
        ]
        reply = groq_chat(final_messages, temperature=0.55, max_tokens=1400).replace("*", "").strip()

        return jsonify({
            "reply": reply,
            "sources": [
                {"title": item["title"], "url": item["url"]}
                for item in sources
            ],
            "searched": searched,
        })

    except requests.Timeout:
        app.logger.exception("Timeout ao contactar um serviço externo.")
        return jsonify({
            "reply": "A minha ligação demorou demasiado desta vez. Tenta novamente daqui a pouco. 🎮",
            "sources": [],
            "searched": searched,
        }), 504
    except Exception:
        app.logger.exception("Erro no endpoint /chat.")
        return jsonify({
            "reply": "Tive um problema técnico ao processar a mensagem. Tenta novamente daqui a pouco.",
            "sources": [],
            "searched": searched,
        }), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))

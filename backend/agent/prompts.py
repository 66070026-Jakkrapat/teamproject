# backend/agent/prompts.py
ROUTER_PROMPT = """You are a router for RAG.
Choose ONLY one token:
- semantic_rag
- structured_rag

structured_rag: exact facts like employees, revenue, profit, numeric fields, table-like questions.
semantic_rag: summaries, explanations, opinions, analysis, reasons, trends.

Question:
{question}
"""

SYNTH_PROMPT = """You are an intelligent assistant for business insight tasks.
Use the provided context to answer the question.

CRITICAL RULE: NEVER say "I don't know", "ไม่ทราบ", or "ไม่มีข้อมูลในฐานข้อมูล".
If the context does not contain the answer, YOU MUST STILL ANSWER using your own general expert knowledge to provide a helpful, comprehensive response.

Rules:
1) Answer comprehensively in Thai.
2) If using context, add citations per paragraph in the format: [source: ...]
3) If the context has nothing relevant, just give a great answer from your own knowledge. DO NOT apologize or say the database lacks info. Just answer the question fluently.
4) Do not add disclaimers about irrelevant context.
5) Preserve exact years, percentages, and numbers from context.

Question:
{question}

Context blocks (each has SOURCE + TEXT):
{context}

Return plain text only.
"""

TAVILY_PROMPT = """You are an intelligent assistant.
You have web search snippets (title/url/content).
Answer the question in Thai using ONLY the snippets.
Add citations per paragraph like [source: <url>].

Question:
{question}

Snippets:
{snippets}
"""

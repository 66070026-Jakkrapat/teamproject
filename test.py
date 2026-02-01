# test.py
from __future__ import annotations

import os
import json
import argparse
import asyncio
import requests

from backend.settings import settings
from backend.rag.rag_store import RAGStore
from backend.evaluation.eval_runner import run_eval

def cmd_eval(args):
    store = RAGStore(settings.DATABASE_URL, settings.OLLAMA_HOST, settings.EMBED_MODEL, settings.EMBED_DIMS)
    async def run():
        await store.init_db()
        res = await run_eval(store, args.dataset, args.namespace, k=args.k)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    asyncio.run(run())

def cmd_ask(args):
    url = f"http://127.0.0.1:{settings.API_PORT}/ask"
    r = requests.post(url, json={"question": args.question, "top_k": args.top_k}, timeout=180)
    print(r.status_code)
    print(r.text)

def parse_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("eval")
    p1.add_argument("--dataset", required=True)
    p1.add_argument("--namespace", default="external")
    p1.add_argument("--k", type=int, default=5)

    p2 = sub.add_parser("ask")
    p2.add_argument("--question", required=True)
    p2.add_argument("--top_k", type=int, default=8)

    return ap.parse_args()

def main():
    args = parse_args()
    if args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "ask":
        cmd_ask(args)

if __name__ == "__main__":
    main()

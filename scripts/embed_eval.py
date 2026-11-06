"""A/B eval of qmx embedding setups on identical index snapshots (only vec_chunks differs).

Setup: ``sqlite3 ~/.qmx/index.db ".backup DIR/a.db"``, copy it to ``DIR/b.db``, write one config per
setup (production config with ``db_path`` and the ``embed_*`` keys changed) and convert the copy
with ``qmx reembed --db DIR/b.db --in-place``. Data and labels live in ``--dir`` (private: they
quote your index); this script holds none.

Steps (``uv run python scripts/embed_eval.py --dir DIR <step> ...``):
  sets      known-item queries written by the judge from sampled chunks + real past qmx queries
  retrieve  run every query through each config: vector-only, RRF (vector+BM25), full (RRF+rerank)
  label     judge relevance (0/1/2) of the pooled full-pipeline top-5 for the real queries
  dedup     nearest-lesson neighbours per model, labelled with the production dedup prompt
  report    metrics + paired bootstrap 95% CIs

The judge (Qwen3.8, the production consolidation model) writes queries and labels, so any family
bias favours the incumbent Qwen embedder. Labels are cached; reruns only ask for what is missing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from qmx.chat import ChatReplyError, make_chat
from qmx.config import Settings
from qmx.consolidate import _lesson_text, judge_config, main_rule
from qmx.embed import EmbedBackendError, embed_queries, make_embedder
from qmx.rerank import make_reranker
from qmx.search import search
from qmx.store import Store

D = Path.home() / ".qmx" / "eval"  # set from --dir
EXCLUDE: list[str] = []  # path substrings to skip (e.g. the session running the eval), --exclude
SEED = 20261105
STRATA = {"learning": 50, "chat": 50, "code": 25, "doc": 25, "meeting": 15, "memory": 10}
WORKERS = 4

GEN_SYSTEM = (
    "You write evaluation queries for a personal search engine over code, notes, meeting "
    "transcripts, chat transcripts and distilled lessons. Given one PASSAGE, imagine the owner "
    "later needs exactly the information in it. Return JSON: usable = false if the passage holds "
    "no specific, findable information (boilerplate, raw logs, bare acknowledgements, tool "
    "noise). keyword_query = 2-6 words the owner would type into a search box to find it. "
    "question = one natural question (8-20 words) whose answer is in the passage, in your own "
    "words: do NOT copy distinctive phrases, identifiers, file names, numbers or rare words from "
    "the passage. Both must be specific enough that this passage is the best answer, not generic."
)
GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "keyword_query": {"type": "string"},
        "question": {"type": "string"},
    },
    "required": ["usable", "keyword_query", "question"],
}
REL_SYSTEM = (
    "You grade search results. The owner of a personal knowledge base (code, notes, meetings, "
    "chat transcripts, lessons) typed QUERY. Rate how useful PASSAGE is for what they were looking "
    "for: 2 = directly answers it or is exactly what they wanted; 1 = on the topic and partly "
    "useful; 0 = not useful. Return JSON {\"rating\": 0|1|2}."
)
REL_SCHEMA = {
    "type": "object",
    "properties": {"rating": {"type": "integer", "enum": [0, 1, 2]}},
    "required": ["rating"],
}


def settings_for(cfg: str) -> Settings:
    return Settings.load(config_path=Path(cfg).expanduser())


def ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def jdump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def jload(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def pmap(fn, items):
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(fn, items))


# -- sets --------------------------------------------------------------------------------------


def cmd_sets(args) -> None:
    s = settings_for(args.judge_config)
    con = ro(s.db_path)
    rng = random.Random(SEED)
    picked: list[tuple[int, str, str, str]] = []
    for kind, n in STRATA.items():
        rows = con.execute(
            """SELECT c.chunk_id, d.kind, d.path, c.text FROM chunks c
               JOIN mentions m ON m.chunk_id = c.chunk_id
               JOIN documents d ON d.doc_id = m.doc_id
               WHERE d.kind = ? AND length(c.text) BETWEEN 300 AND 6000
               GROUP BY c.chunk_id""",
            (kind,),
        ).fetchall()
        rows = [r for r in rows if not any(x in (r[2] or "") for x in EXCLUDE)]
        if kind == "learning":  # live lessons only
            live = {r[0] for r in con.execute(
                """SELECT m.chunk_id FROM learnings l JOIN mentions m ON m.doc_id = l.doc_id
                   WHERE l.deprecated_at IS NULL AND l.superseded_by IS NULL""")}
            rows = [r for r in rows if r[0] in live]
        picked += rng.sample(rows, min(n, len(rows)))
    chat = make_chat(s)

    def gen(row):
        cid, kind, path, text = row
        try:
            r = chat.complete_json(GEN_SYSTEM, f"PASSAGE ({kind}):\n{text[:3000]}",
                                   schema=GEN_SCHEMA, max_tokens=400)
        except ChatReplyError as exc:
            return {"chunk_id": cid, "kind": kind, "error": str(exc)}
        return {"chunk_id": cid, "kind": kind, "path": path, **r}

    t = time.time()
    out = pmap(gen, picked)
    known = []
    for r in out:
        if r.get("usable") and r.get("keyword_query") and r.get("question"):
            for style in ("keyword_query", "question"):
                known.append({"qid": f"{r['chunk_id']}:{style}", "style": style, "kind": r["kind"],
                              "target": r["chunk_id"], "query": r[style].strip()})
    jdump(D / "known_raw.jsonl", out)
    jdump(D / "known.jsonl", known)
    print(f"known-item: {len(picked)} chunks -> {len(known)} queries "
          f"({sum(1 for r in out if not r.get('usable'))} unusable/err) in {time.time()-t:.0f}s")

    seen, real = set(), []
    for f in glob.glob(os.path.expanduser("~/.claude/projects/**/*.jsonl"), recursive=True):
        if any(x in f for x in EXCLUDE):
            continue
        with open(f, errors="ignore") as fh:
            for line in fh:
                if "mcp__qmx__query" not in line:
                    continue
                try:
                    msg = json.loads(line).get("message", {})
                except json.JSONDecodeError:
                    continue
                for c in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
                    if c.get("type") == "tool_use" and c.get("name") == "mcp__qmx__query":
                        q = str(c.get("input", {}).get("text", "")).strip()
                        if len(q.split()) >= 3 and q.lower() not in seen:
                            seen.add(q.lower())
                            real.append(q)
    real.sort()
    real = random.Random(SEED).sample(real, min(args.real, len(real)))
    jdump(D / "real.jsonl", [{"qid": f"real{i}", "query": q} for i, q in enumerate(real)])
    print(f"real queries: {len(seen)} distinct found, sampled {len(real)}")


# -- retrieve ----------------------------------------------------------------------------------


class Strict:
    """Embedder wrapper that counts failures (search() would silently degrade to BM25)."""

    def __init__(self, inner):
        self.inner, self.failures, self.secs = inner, 0, 0.0

    @property
    def dim(self):
        return self.inner.dim

    def embed(self, texts):
        return self.inner.embed(texts)

    def embed_query(self, texts):
        t = time.perf_counter()
        try:
            return embed_queries(self.inner, texts)
        except EmbedBackendError:
            self.failures += 1
            raise
        finally:
            self.secs += time.perf_counter() - t


def rank_of(target: int, hits) -> int | None:
    for i, h in enumerate(hits, 1):
        if (h.hit.chunk_id if hasattr(h, "hit") else h.chunk_id) == target:
            return i
    return None


def cmd_retrieve(args) -> None:
    known, real = jload(D / "known.jsonl"), jload(D / "real.jsonl")
    for cfg in args.configs:
        s = settings_for(cfg)
        name = Path(cfg).stem
        emb = Strict(make_embedder(s))
        rr = make_reranker(s)
        rr_fail = 0
        out = []
        vec_secs = 0.0
        with Store.open(s.db_path, s.embed_dim, s.embed_model) as store:
            for q in known + real:
                row = {"qid": q["qid"]}
                [qv] = emb.embed_query([q["query"]])
                t = time.perf_counter()
                vhits = store.search_vec(qv, k=20)
                vec_secs += time.perf_counter() - t
                rrf = search(store, emb, q["query"], k=5)
                full = search(store, emb, q["query"], k=5, reranker=rr)
                if rr is not None and getattr(rr, "_open_until", 0) > 0:
                    rr_fail += 1
                if "target" in q:
                    row.update(vec=rank_of(q["target"], vhits), rrf=rank_of(q["target"], rrf),
                               full=rank_of(q["target"], full))
                row["full_top5"] = [h.hit.chunk_id for h in full]
                out.append(row)
        jdump(D / f"ret_{name}.jsonl", out)
        n = len(known) + len(real)
        print(f"{name}: {n} queries; embed failures {emb.failures}, rerank failures {rr_fail}; "
              f"mean query-embed {1000*emb.secs/max(1, 3*n):.0f} ms, "
              f"mean vector search {1000*vec_secs/max(1, n):.0f} ms")
        if emb.failures or rr_fail:
            sys.exit(f"{name}: backend failures -> results not trustworthy, rerun")


# -- label (real queries) ----------------------------------------------------------------------


def cmd_label(args) -> None:
    s = settings_for(args.judge_config)
    real = {q["qid"]: q["query"] for q in jload(D / "real.jsonl")}
    pool: set[tuple[str, int]] = set()
    for cfg in args.configs:
        for r in jload(D / f"ret_{Path(cfg).stem}.jsonl"):
            if r["qid"] in real:
                pool |= {(r["qid"], c) for c in r["full_top5"]}
    cache_path = D / "rel_labels.jsonl"
    cache = {(r["qid"], r["chunk_id"]): r["rating"] for r in jload(cache_path)}
    todo = sorted(pool - cache.keys())
    con = ro(s.db_path)
    text = {cid: t for cid, t in con.execute(
        f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({','.join('?'*len(pool))})",
        [c for _, c in pool])} if pool else {}
    chat = make_chat(s)

    def grade(key):
        qid, cid = key
        try:
            user = f"QUERY: {real[qid]}\n\nPASSAGE:\n{text[cid][:2000]}"
            r = chat.complete_json(REL_SYSTEM, user, schema=REL_SCHEMA, max_tokens=60)
            return {"qid": qid, "chunk_id": cid, "rating": int(r["rating"])}
        except (ChatReplyError, KeyError, ValueError, TypeError):
            return None

    t = time.time()
    new = [r for r in pmap(grade, todo) if r]
    with cache_path.open("a") as fh:
        for r in new:
            fh.write(json.dumps(r) + "\n")
    print(f"relevance: pool {len(pool)}, cached {len(cache)}, labelled {len(new)}/{len(todo)} "
          f"in {time.time()-t:.0f}s")


# -- dedup -------------------------------------------------------------------------------------


def lesson_vectors(db: Path) -> tuple[list[int], dict[int, tuple[str, str | None]], np.ndarray]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    from qmx.store import load_sqlite_vec

    load_sqlite_vec(con)
    rows = con.execute(
        """SELECT l.learning_id, l.statement, l.detail, m.chunk_id FROM learnings l
           JOIN mentions m ON m.doc_id = l.doc_id
           WHERE l.deprecated_at IS NULL AND l.superseded_by IS NULL
           ORDER BY l.learning_id""").fetchall()
    ids, info, vecs = [], {}, []
    for lid, st, de, cid in rows:
        blob = con.execute("SELECT embedding FROM vec_chunks WHERE chunk_id=?", (cid,)).fetchone()
        if blob is None:
            continue
        v = np.frombuffer(blob[0], dtype=np.float32)
        ids.append(lid)
        info[lid] = (st, de)
        vecs.append(v / (np.linalg.norm(v) or 1.0))
    return ids, info, np.vstack(vecs)


def cmd_dedup(args) -> None:
    s = settings_for(args.judge_config)
    models = {Path(c).stem: lesson_vectors(settings_for(c).db_path) for c in args.configs}
    common = set.intersection(*(set(m[0]) for m in models.values()))
    any_info = next(iter(models.values()))[1]
    anchors = random.Random(SEED).sample(sorted(common), args.anchors)
    neigh: dict[str, set[tuple[int, int]]] = {}
    for name, (ids, _, M) in models.items():
        idx = {lid: i for i, lid in enumerate(ids)}
        pairs = set()
        for a in anchors:
            sims = M @ M[idx[a]]
            sims[idx[a]] = -2
            for j in np.argsort(-sims)[: args.top]:
                pairs.add((a, ids[j]))
        neigh[name] = pairs
    union = set().union(*neigh.values())
    cache_path = D / "dedup_labels.jsonl"
    cache = {(r["a"], r["b"]): r["same"] for r in jload(cache_path)}
    todo = sorted(union - cache.keys())
    j = judge_config(s)
    chat = make_chat(s)

    def label(pair):
        a, b = pair  # b plays the stored lesson, a the new candidate (as at insert time)
        ex = _lesson_text(*any_info[b])
        cand = _lesson_text(*any_info[a])
        user = (f"EXISTING:\n{ex}\n\nEXISTING MAIN RULE:\n{main_rule(any_info[b][0])}"
                f"\n\nCANDIDATE:\n{cand}") if j.v2 else f"EXISTING:\n{ex}\n\nCANDIDATE:\n{cand}"
        try:
            r = chat.complete_json(j.prompts.dedup_system, user, schema=j.prompts.dedup_schema,
                                   max_tokens=j.short_max_tokens)
            return {"a": a, "b": b, "same": r.get("same_lesson") is True}
        except ChatReplyError:
            return None

    t = time.time()
    new = [r for r in pmap(label, todo) if r]
    with cache_path.open("a") as fh:
        for r in new:
            fh.write(json.dumps(r) + "\n")
    print(f"dedup: {len(anchors)} anchors, union {len(union)} pairs "
          + ", ".join(f"{k} {len(v)}" for k, v in neigh.items())
          + f"; labelled {len(new)}/{len(todo)} in {time.time()-t:.0f}s")
    jdump(D / "dedup_pairs.jsonl", [{"model": k, "a": a, "b": b} for k, v in neigh.items()
                                    for a, b in sorted(v)])


# -- report ------------------------------------------------------------------------------------


def boot_ci(diff: np.ndarray, n: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def rr(rank, cutoff):
    return 0.0 if rank is None or rank > cutoff else 1.0 / rank


def hit(rank, cutoff):
    return 1.0 if rank is not None and rank <= cutoff else 0.0


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    p, q = np.array(pos)[:, None], np.array(neg)[None, :]
    return float((p > q).mean() + 0.5 * (p == q).mean())


def cmd_report(args) -> None:  # noqa: C901 (one linear report)
    names = [Path(c).stem for c in args.configs]
    base = names[0]
    known = {q["qid"]: q for q in jload(D / "known.jsonl")}
    ret = {n: {r["qid"]: r for r in jload(D / f"ret_{n}.jsonl")} for n in names}
    qids = sorted(set(known) & set.intersection(*(set(r) for r in ret.values())))
    print(f"## Known-item retrieval ({len(qids)} queries; target = the chunk the query was written "
          "from)\n")
    print("| setup | arm | Recall@1 | Recall@5 | MRR@5 | Recall@20 (vector) |")
    print("|---|---|---|---|---|---|")
    per = {}
    for n in names:
        for arm in ("vec", "rrf", "full"):
            ranks = [ret[n][q][arm] for q in qids]
            per[(n, arm)] = (
                np.array([rr(r, 5) for r in ranks]),
                np.array([hit(r, 5) for r in ranks]),
            )
            r20 = f"{np.mean([hit(r, 20) for r in ranks]):.3f}" if arm == "vec" else ""
            print(f"| {n} | {arm} | {np.mean([hit(r, 1) for r in ranks]):.3f} | "
                  f"{per[(n, arm)][1].mean():.3f} | {per[(n, arm)][0].mean():.3f} | {r20} |")
    print("\nPaired differences vs", base, "(mean, bootstrap 95% CI):\n")
    for n in names[1:]:
        for arm in ("vec", "rrf", "full"):
            for label, k in (("MRR@5", 0), ("Recall@5", 1)):
                d = per[(n, arm)][k] - per[(base, arm)][k]
                lo, hi = boot_ci(d)
                flag = "BETTER" if lo > 0 else ("WORSE" if hi < 0 else "n.s.")
                print(f"- {n} {arm} {label}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {flag}")
    print("\nFull-pipeline MRR@5 by kind / style:\n")
    groups: dict[str, list[str]] = {}
    for q in qids:
        groups.setdefault(known[q]["kind"], []).append(q)
        groups.setdefault(known[q]["style"], []).append(q)
    print("| group | n | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for g, qs in sorted(groups.items()):
        vals = [np.mean([rr(ret[n][q]["full"], 5) for q in qs]) for n in names]
        print(f"| {g} | {len(qs)} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

    labels = {(r["qid"], r["chunk_id"]): r["rating"] for r in jload(D / "rel_labels.jsonl")}
    real = [q["qid"] for q in jload(D / "real.jsonl")]
    if labels and real:
        def ndcg(qid, n):
            got = [labels.get((qid, c), 0) for c in ret[n][qid]["full_top5"]]
            pool = sorted((v for (q, _), v in labels.items() if q == qid), reverse=True)[:5]
            dcg = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(got))
            idcg = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(pool))
            return dcg / idcg if idcg else 0.0

        def p2(qid, n):
            return np.mean([labels.get((qid, c), 0) == 2 for c in ret[n][qid]["full_top5"]])

        print(f"\n## Real past queries ({len(real)}; full pipeline top-5, judge-graded 0/1/2, "
              "pooled)\n")
        print("| setup | nDCG@5 | share of top-5 graded 2 |")
        print("|---|---|---|")
        vals = {n: (np.array([ndcg(q, n) for q in real]), np.array([p2(q, n) for q in real]))
                for n in names}
        for n in names:
            print(f"| {n} | {vals[n][0].mean():.3f} | {vals[n][1].mean():.3f} |")
        for n in names[1:]:
            d = vals[n][0] - vals[base][0]
            lo, hi = boot_ci(d)
            print(f"\n- {n} nDCG@5 vs {base}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

    dl = {(r["a"], r["b"]): r["same"] for r in jload(D / "dedup_labels.jsonl")}
    if dl:
        pairs = jload(D / "dedup_pairs.jsonl")
        positives = {k for k, v in dl.items() if v}
        print(f"\n## Duplicate lessons ({len(dl)} judge-labelled neighbour pairs, "
              f"{len(positives)} judged the same lesson)\n")
        print("| setup | duplicates found in its top neighbours | AUC (cosine separates same vs "
              "different) | look-alikes passing at the cut that keeps 90% of duplicates | "
              "cut |")
        print("|---|---|---|---|---|")
        for n in names:
            ids, _, M = lesson_vectors(settings_for(next(c for c in args.configs
                                                        if Path(c).stem == n)).db_path)
            idx = {lid: i for i, lid in enumerate(ids)}
            found = {(p["a"], p["b"]) for p in pairs if p["model"] == n} & positives
            sims = {
                k: float(M[idx[k[0]]] @ M[idx[k[1]]]) for k in dl if k[0] in idx and k[1] in idx
            }
            pos = [v for k, v in sims.items() if dl[k]]
            neg = [v for k, v in sims.items() if not dl[k]]
            cut = float(np.percentile(pos, 10)) if pos else float("nan")
            passing = np.mean([v >= cut for v in neg]) if neg else float("nan")
            print(f"| {n} | {len(found)}/{len(positives)} | {auc(pos, neg):.3f} | "
                  f"{passing:.1%} | {cut:.3f} |")


def main() -> None:
    global D
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(D), help="eval data dir (snapshots' labels, results)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="skip sources whose path contains this (repeatable)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sets")
    p.add_argument("--judge-config", required=True)
    p.add_argument("--real", type=int, default=40)
    p = sub.add_parser("retrieve")
    p.add_argument("configs", nargs="+")
    p = sub.add_parser("label")
    p.add_argument("--judge-config", required=True)
    p.add_argument("configs", nargs="+")
    p = sub.add_parser("dedup")
    p.add_argument("--judge-config", required=True)
    p.add_argument("--anchors", type=int, default=100)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("configs", nargs="+")
    p = sub.add_parser("report")
    p.add_argument("configs", nargs="+")
    a = ap.parse_args()
    D = Path(a.dir).expanduser()
    D.mkdir(parents=True, exist_ok=True)
    EXCLUDE.extend(a.exclude)
    {"sets": cmd_sets, "retrieve": cmd_retrieve, "label": cmd_label, "dedup": cmd_dedup,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()

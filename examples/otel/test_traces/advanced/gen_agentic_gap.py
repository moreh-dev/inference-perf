"""Generate a long agentic "gap-reuse" OTel trace for KV-cache retention testing.

Structure: one agentic session where a coordinator ("main") agent reuses a large
shared system prefix P every round, interleaved with independent "tool" spans whose
prefixes differ (churn). The tool spans do NOT touch P, so under a tight KV cache P
goes stale and LRU evicts it between main reuses; a retention policy that protects P
keeps each main span's prefill cheap. This exercises the prefill-bound TTFT benefit
of retention even at N=1, C=1 (which the short software_architecture_review trace
cannot, since its prefill is a negligible fraction of e2e).

Outputs are forced short via gen_ai.usage.completion_tokens so the workload stays
prefill-bound. Timestamps are ISO strings because the replay-graph builder parses
them with datetime.fromisoformat (a numeric start_time raises and yields 0 sessions).

Run:  python gen_agentic_gap.py   ->  writes agentic_gap_temp0.json next to this file
(deterministic: same output every run). For a >1MB ConfigMap mount, gzip the result
and gunzip it back in an init container.
"""
import json
import os
from datetime import datetime, timedelta

# ---- params (60 turns = 15 main + 45 tool; real ~84.8k tok, ~99k chars/4) ----
P_TOK = 76000       # shared high-value prefix (reused by every main span)
R = 15              # rounds (= main spans)
T = 3               # tool (churn) spans per round; main+tool = 60 spans
TOOL_TOK = 6000     # per-tool input tokens (independent prefix = churn)
Q_TOK = 1500        # per-round main-agent query tokens (accumulating)
OUT_TOK = 32        # forced output length -> prefill-bound
CPT = 4             # approx chars per token (sizing only)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentic_gap_temp0.json")
_BASE = datetime(2026, 3, 17, 10, 0, 0)


def _iso(sec):
    return (_BASE + timedelta(seconds=sec)).isoformat(timespec="microseconds")


def filler(ntok, tag):
    s = (f"[{tag}] distributed-systems spec fragment {tag}: consistency, "
         f"partitioning, replication, failover, latency, throughput across regions. ")
    need = ntok * CPT
    return (s * (need // len(s) + 1))[:need]


def make_span(idx, sid, msgs, out):
    return {
        "span_id": sid,
        "parent_span_id": None,
        "name": "chat gpt-4",
        "kind": "CLIENT",
        "start_time": _iso(idx * 2),
        "end_time": _iso(idx * 2 + 1),
        "status": {"status_code": "OK"},
        "trace_id": "agentic_gap_001",
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "gpt-4",
            "gen_ai.request.temperature": 0,
            "gen_ai.request.max_tokens": 4096,
            "gen_ai.input.messages": json.dumps(msgs),
            "gen_ai.output.text": out,
            "gen_ai.usage.completion_tokens": OUT_TOK,
            "exgentic.session.id": "sess_agentic_gap",
        },
        "resource_attributes": {"service.name": "agentic-replay"},
    }


def main():
    prefix = ("SYSTEM: You are the lead orchestrator agent with the following tool "
              "definitions and reference corpus (reused every round).\n\n"
              + filler(P_TOK, "SHARED_P"))
    spans = []
    idx = 0
    hist = [{"role": "system", "content": prefix}]
    for r in range(1, R + 1):
        hist.append({"role": "user",
                     "content": f"Round {r}: " + filler(Q_TOK, f"MAINQ{r:02d}")
                     + f" Synthesize so far and plan round {r}."})
        idx += 1
        ans = f"Round {r}: plan set; dispatching {T} tools."
        spans.append(make_span(idx, f"span_{idx:03d}_main_r{r:02d}", list(hist), ans))
        hist.append({"role": "assistant", "content": ans})
        for t in range(1, T + 1):
            tool = [
                {"role": "system",
                 "content": f"You are specialist tool #{t} (round {r}). Independent context."},
                {"role": "user",
                 "content": filler(TOOL_TOK, f"TOOL_r{r:02d}_t{t}") + f" Analyze input for tool {t}."},
            ]
            idx += 1
            spans.append(make_span(idx, f"span_{idx:03d}_tool_r{r:02d}_t{t}", tool,
                                   f"Tool {t} round {r} result."))
    trace = {"trace_id": "agentic_gap_001", "span_count": len(spans),
             "collected_at": "2026-01-01T00:00:00Z", "spans": spans}
    with open(OUT, "w") as f:
        json.dump(trace, f)
    mains = [s for s in spans if "_main_" in s["span_id"]]
    last = len(mains[-1]["attributes"]["gen_ai.input.messages"]) // CPT
    print(f"wrote {OUT}")
    print(f"spans={len(spans)} (main={len(mains)}, tool={len(spans) - len(mains)}), "
          f"P~{P_TOK}t, churn~{T}x{TOOL_TOK}={T * TOOL_TOK}t/round, max main prompt~{last}t (chars/4)")


if __name__ == "__main__":
    main()

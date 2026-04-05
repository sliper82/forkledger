"""ForkLedger benchmark — decision retrieval accuracy and latency."""
import argparse, random, tempfile, time, uuid, sys
from pathlib import Path
from statistics import median, quantiles

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from forkledger import ForkLedgerEngine, ForkRecord, Branch, OutcomeEstimate

STATES = [
    {"task": "research", "signal": "mixed",    "deadline": "tight"},
    {"task": "research", "signal": "mixed",    "deadline": "normal"},
    {"task": "trade",    "signal": "bullish",  "volume": "high"},
    {"task": "trade",    "signal": "bearish",  "volume": "low"},
    {"task": "code",     "signal": "ambiguous","priority": "high"},
]
BRANCHES = {
    "research": (["verify","fast-publish","wait"], "verify"),
    "trade":    (["enter","wait","short"],          "enter"),
    "code":     (["refactor","quick-fix","skip"],   "refactor"),
}

def make_record(state):
    task = state.get("task", "research")
    branches, best = BRANCHES.get(task, (["a","b"], "a"))
    chosen = best if random.random() > 0.3 else random.choice(branches)
    realized = (1.5 if chosen == best else -0.5) + random.gauss(0, 0.3)
    estimated = [OutcomeEstimate(branch_name=b,
        estimated_value=(1.2 if b==best else -0.3)+random.gauss(0,0.2))
        for b in branches if b != chosen]
    return ForkRecord(
        fork_id=str(uuid.uuid4()), pre_state=dict(state),
        trigger=f"bench-{task}",
        possible_branches=[Branch(name=b) for b in branches],
        chosen_branch=chosen, realized_value=round(realized,3),
        estimated_outcomes=estimated,
        confidence=round(random.uniform(0.6,1.0),2), tags=[task],
    )

def run(n_records=200, n_queries=50):
    print(f"\nForkLedger Benchmark — {n_records} records, {n_queries} queries")
    print("="*60)
    with tempfile.TemporaryDirectory() as tmp:
        engine = ForkLedgerEngine(f"{tmp}/bench.db", backend="sqlite")
        t0 = time.perf_counter()
        for _ in range(n_records):
            engine.add_record(make_record(random.choice(STATES)))
        lt = time.perf_counter()-t0
        print(f"Load: {lt:.2f}s ({n_records/lt:.0f} rec/s)")
        lats, hits = [], 0
        for _ in range(n_queries):
            state = random.choice(STATES)
            task = state.get("task","research")
            _, best = BRANCHES.get(task,([],"a"))
            t0 = time.perf_counter()
            recs = engine.recommend(state, top_k=3)
            lats.append((time.perf_counter()-t0)*1000)
            if recs and recs[0]["branch"]==best: hits+=1
        p50=median(lats); p95=quantiles(lats,n=20)[18]
        print(f"Latency: p50={p50:.1f}ms  p95={p95:.1f}ms")
        print(f"Precision@1: {hits/n_queries:.1%}")
        pols=engine.policies(min_support=3,fuzzy=True)
        ok=sum(1 for p in pols if p["state"].get("task") in BRANCHES and
               p["recommended_branch"]==BRANCHES[p["state"]["task"]][1])
        if pols: print(f"Policy accuracy: {ok}/{len(pols)} = {ok/len(pols):.1%}")
        print("="*60)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--records",type=int,default=200)
    ap.add_argument("--queries",type=int,default=50)
    a=ap.parse_args(); run(a.records,a.queries)

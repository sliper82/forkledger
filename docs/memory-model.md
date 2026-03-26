# Memory model

A ForkLedger record is designed to capture the smallest useful unit of decision memory.

## Record schema

```json
{
  "fork_id": "unique-id",
  "pre_state": {},
  "trigger": "what made this matter",
  "possible_branches": [],
  "chosen_branch": "selected option",
  "realized_value": 0.0,
  "estimated_outcomes": [],
  "regret_vector": {},
  "constraints": {},
  "provenance": {},
  "confidence": 0.5,
  "expiry": null,
  "created_at": "ISO-8601 timestamp",
  "tags": []
}
```

## Semantics

### pre_state
A compact representation of the world before commitment.

### trigger
The reason this point deserves memory. It should explain why the choice was consequential.

### possible_branches
The set of options available at the decision point.

### chosen_branch
The option that was actually executed.

### realized_value
The observed utility of the chosen branch.

### estimated_outcomes
Counterfactual estimates for non-chosen branches.

### regret_vector
The opportunity cost of each branch relative to the best branch value in the record.

### constraints
Rules, limits, and hard boundaries active at the time.

### provenance
Where the record came from and how trustworthy it is.

### confidence
How much the system trusts the record as evidence.

### expiry
Optional invalidation boundary for stale memories.

## Retrieval philosophy

ForkLedger is not optimized for pure semantic recall.
It is optimized for **decision reuse**.

The goal is not to ask:
- what sounds similar?

The goal is to ask:
- what past fork most resembles this choice situation?
- which branch repeatedly led to lower regret?
- what should the system avoid doing again?

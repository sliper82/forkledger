# Architecture

ForkLedger is organized around a single principle:

> Store decision structure, not just conversational residue.

## Components

### 1. Fork record model
The system stores a structured representation of a meaningful decision point.

Each record includes:

- pre-state
- trigger
- possible branches
- chosen branch
- realized value
- estimated alternative outcomes
- regret vector
- constraints
- provenance
- confidence
- expiry

### 2. Counterfactual regret layer
Estimated branch outcomes are converted into a regret vector.

This allows the system to compare not only what happened, but what the choice cost relative to alternatives.

### 3. Retrieval layer
Records are ranked using:

- state overlap
- constraint overlap
- confidence
- recency
- regret salience

This can be replaced later with learned encoders or hybrid retrieval systems.

### 4. Policy distillation layer
When repeated states appear, ForkLedger groups them and computes average regret by branch.

That produces a lightweight policy surface:
- under this state pattern
- with this level of support
- this branch has historically produced the lowest regret

## Why this differs from ordinary memory

Typical AI memory stores:
- messages
- chunks
- embeddings
- facts
- graph links

ForkLedger stores:
- decision geometry
- alternative paths
- regret signals
- branch-level support

That makes it directly useful for action selection and audit.

## Extension points

- learned state encoders
- vector-assisted retrieval
- simulator-backed counterfactuals
- multi-agent shared memory
- evaluation dashboards
- time-decay policies
- domain-specific branch scorers

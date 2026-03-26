# ForkLedger

**ForkLedger** is a branch-based memory engine for AI systems.

Instead of treating memory as a list of messages, chunks, or embeddings, ForkLedger stores decision points: the state before a choice, the available branches, the action that was taken, the observed result, and the estimated regret of alternatives.

This repository provides a working reference implementation of that model for researchers, builders, agent developers, and AI infrastructure teams.

## Why ForkLedger exists

Most memory systems answer questions like:

- What happened?
- What was said?
- What facts seem similar to this query?

ForkLedger answers a different question:

- When this situation appeared before, what options existed, which path was chosen, and what did that choice cost?

That makes it useful for systems that need to improve decisions over time, not just retrieve context.

## Core idea

A memory unit in ForkLedger is a fork.

Each fork captures:

- the pre-decision state
- the trigger that made the decision important
- the available branches
- the chosen branch
- the observed outcome
- estimated outcomes for non-chosen branches
- a regret vector
- constraints and provenance
- a confidence score and expiry policy

Over time, the engine can retrieve similar historical forks and rank the branches that previously led to better outcomes under similar conditions.

## What is included

- A structured Python implementation of the ForkLedger memory model
- JSON-backed storage
- A branch scoring and retrieval engine
- Counterfactual regret estimation helpers
- A simple policy distillation mechanism
- CLI tools for creating, listing, querying, and summarizing fork records
- Example data and tests

## Support and project direction

ForkLedger is being built in the open. Early users, contributors, testers, and backers can help shape the direction of the project.

Support helps accelerate work on:

- evaluation infrastructure
- better simulators
- dashboards and observability
- integrations with real agent runtimes
- stronger policy learning loops

A direct funding link can be added to this repository as soon as a public sponsorship URL is configured.

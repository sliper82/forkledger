#!/usr/bin/env bash
# ForkLedger REST API demo
# Requires: pip install forkledger[api]
# Run server first: forkledger serve --backend sqlite --store /tmp/demo.db

BASE="http://localhost:8000"

echo "=== ForkLedger API Demo ==="

echo -e "\n[health]"
curl -s $BASE/health | python3 -m json.tool

echo -e "\n[add fork]"
curl -s -X POST $BASE/forks \
  -H "Content-Type: application/json" \
  -d '{
    "fork_id": "api-test-001",
    "pre_state": {"task": "research", "signal": "conflicted"},
    "trigger": "Sources contradict",
    "possible_branches": [{"name": "verify"}, {"name": "publish"}],
    "chosen_branch": "verify",
    "realized_value": 1.5,
    "confidence": 0.8,
    "tags": ["test"]
  }' | python3 -m json.tool

echo -e "\n[recommend]"
curl -s -X POST $BASE/recommend \
  -H "Content-Type: application/json" \
  -d '{"current_state": {"task": "research", "signal": "conflicted"}, "top_k": 3}' \
  | python3 -m json.tool

echo -e "\n[stats]"
curl -s $BASE/stats | python3 -m json.tool

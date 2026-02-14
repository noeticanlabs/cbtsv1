---
title: "Algorithms (Canonical)"
description: "Canonical algorithms including Banker Algorithm for accept/retry/abort and Repair/Projection algorithms"
last_updated: "2026-02-07"
authors: ["NoeticanLabs"]
tags: ["coherence", "algorithms", "canonical", "implementation"]
---

# 08 Algorithms (Canonical)

## Rail Priority Policy (default)
1. Reduce step size / \(\Delta t\)
2. Increase damping or regularization
3. Project/repair
4. Fetch evidence / tool call
5. Backtrack to checkpoint

## Banker Algorithm (accept/retry/abort)
**Inputs:** current state \(x\), policy \(P\), rails \(R\), retry limit.

**Outputs:** decision, receipt.

**Pseudocode:**
```
function step_with_coherence(x, P, R):
  proposal = propose(x)
  residuals, C = evaluate(proposal)
  gates = check_gates(residuals, C, P)
  if gates.hard_fail:
    receipt = emit_receipt(x, proposal, residuals, C, gates, actions=[])
    return abort, receipt
  if gates.soft_fail:
    for rail in priority_order(R):
      if rail.applicable(gates):
        proposal = rail.apply(proposal)
        residuals, C = evaluate(proposal)
        gates = check_gates(residuals, C, P)
        log_action(rail)
        if gates.pass:
          receipt = emit_receipt(x, proposal, residuals, C, gates, actions)
          return accept, receipt
    receipt = emit_receipt(x, proposal, residuals, C, gates, actions)
    return retry, receipt
  receipt = emit_receipt(x, proposal, residuals, C, gates, actions=[])
  return accept, receipt
```

## Repair / Projection Algorithm
**Inputs:** proposal state \(x'\), constraint set \(\Omega\).

**Outputs:** repaired state, action log.

**Pseudocode:**
```
function repair(x', Omega):
  x_r = project_to_feasible_set(x', Omega)
  return x_r, {rail:"projection", bounds:Omega.bounds}
```

## Backtracking / Checkpoint Algorithm
**Inputs:** checkpoints \(\{x_k\}\), current proposal.

**Outputs:** restored state.

**Pseudocode:**
```
function backtrack(checkpoints):
  x_prev = last_valid(checkpoints)
  return x_prev, {rail:"backtrack", depth:1}
```

## Evidence Escalation Algorithm
**Inputs:** proposal, evidence tools, max budget.

**Outputs:** enriched proposal, tool receipts.

**Pseudocode:**
```
function fetch_evidence(proposal, tools, budget):
  for tool in tools:
    if budget_exhausted(): break
    proposal = tool.enrich(proposal)
    log_action(tool)
  return proposal
```

## HAAI Ladder (optional specialization)
**Inputs:** policy ladder \(L\) of increasing evidence/mitigation.

**Outputs:** accept/retry with receipts.

**Pseudocode:**
```
function haai_step(x, L):
  for rung in L:
    proposal = rung.apply(x)
    if check_gates(...).pass:
      return accept, receipt
  return retry, receipt
```

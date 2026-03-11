# Shared Representation Companion Micro-Demo

This deterministic Python CLI micro-demo shows that identical representational input does not guarantee identical downstream meaning. Each agent receives the same interaction-derived `ExperienceEntry` and the same typed `TVS` representation layer, but interpretation diverges when memory/inference context diverges and converges when memory/inference context aligns.

## Companion Paper

This repository is a deterministic companion micro-demo for the associated paper. The paper provides the broader architectural framing; this repository isolates and demonstrates one narrow claim:

**identical structured representation does not guarantee identical meaning across agents**

Paper:
https://zenodo.org/records/18943706

## Architecture Diagram

```text
WORLD_OBJECT_A
      ↓
Experience Entry
      ↓
Shared TVS Representation
      ↓
Agent-Specific Memory / Inference Context
      ↓
Interpretation Output per Agent
      ↓
Interpretation Divergence Matrix
```

## File Overview
- `main.py`: Runs Run 1 and Run 2, prints shared representation, interpretations, divergence matrix, and comparison summary.
- `models.py`: Dataclasses for `TVS`, `ExperienceEntry`, and `Interpretation`.
- `agents.py`: Deterministic agent interpretation logic for NPC_A, NPC_B, and NPC_C.
- `divergence.py`: Euclidean pairwise interpretation-distance matrix and average pairwise divergence.
- `README.md`: Project framing and usage.
- `LICENSE`: Creative Commons Attribution-NonCommercial 4.0 International license.

## How to Run

```bash
python3 main.py
```

## What This Demo Proves
- Shared representation can remain identical across agents and runs.
- Interpretation is a downstream agent-relative layer shaped by memory/inference context.
- Interpretation divergence is measurable even when representation is identical.
- A meta-observer can model second-order interpretive divergence/convergence across agents.

## Run 1 vs Run 2
- Run 1 applies identical representation with divergent memory context, producing higher interpretation divergence.
- Run 2 applies identical representation with aligned memory context, producing lower interpretation divergence.

## Representation vs Interpretation Boundary
`TVS` in this demo is the shared representation layer, not belief or meaning. `ExperienceEntry` is the delivered event record upstream of interpretation. Interpretation outputs are downstream and agent-relative. Therefore, divergence reported by the matrix is interpretation divergence, not perception divergence.

## Scope Boundary
This repository is a simplified companion micro-demo aligned with architecture papers. It is not a full TVS implementation, full EEC implementation, governance harness, conformance suite, memory promotion engine, field dynamics simulator, or normative proof/break harness.

## Contact
Stephen A. Putman  
Email: putmanmodel@pm.me  
GitHub: https://github.com/putmanmodel

## License
Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. See [LICENSE](LICENSE).

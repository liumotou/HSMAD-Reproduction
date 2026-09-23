# GAT baseline

Status: **Candidate A smoke implementation only — no formal GAT run is authorized.**

The source audit is in [audit/gat_source_protocol_audit.md](audit/gat_source_protocol_audit.md). The fixed official reference is Veličković et al., *Graph Attention Networks*, ICLR 2018, with repository `https://github.com/PetarV-/GAT` at commit `5af87e7fce2b90ae1cbd621cd58059036a3c7436`.

The selected smoke candidate is `protocol_v1_original_gat_8x8_hidden64`: eight
8-dimensional hidden heads are concatenated to 64, followed by a single
two-logit output head.  It is a DGL, auditable implementation of the original
GAT topology, not the original TensorFlow code.  The smoke uses five epochs
only and is never included in a formal summary.

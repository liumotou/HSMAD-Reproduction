# AMNet-HSMAD adapted candidate

This isolated candidate traces its model topology and Bernstein filters to
Illyasville/AMNet commit `74f6a826ddb56dc73f82aa523bf350a1e28b8f03`.

It is marked `candidate_protocol_not_author_exact`. The official model topology,
`K=2`, `M=5`, `beta=1`, and classifier dropout `0.3` are retained. HSMAD overrides
the hidden width to 64 and both filter/non-filter learning rates to 0.01. The
filter group preserves the official no-weight-decay behavior; the non-filter
group uses the frozen project value 0.0001.

The official entry point is not reused because it can create splits and evaluates
the test mask during validation improvements. This adapter requires persisted
HSMAD masks, selects checkpoints only by validation AUPRC, selects the F1 threshold
only on validation, and evaluates test only after both choices are fixed.

The DGL-to-PyG adapter is format-only: it does not add/remove/reverse edges or
change features, labels, or masks. Model `forward` accepts only `x` and
`edge_index`; the marginal constraint is computed outside the model from the
training mask.

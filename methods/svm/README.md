# SVM — HSMAD adapted candidate

This is a feature-only SVM candidate. The model source is the official
scikit-learn SVM API, not an HSMAD author implementation. It never reads graph
edges. Frozen HSMAD masks and the project validation-threshold protocol are
applied outside the classifier.

The current implementation choice is `SVC(kernel="rbf", C=1.0,
class_weight="balanced", probability=True)`. Kernel, C, class weighting and
probability calibration are not published by HSMAD; they are explicitly
project adaptation choices and must not be called author-exact settings.

Status: `candidate_protocol_not_author_exact`. No formal experiment has been
started by this source/TDD step.

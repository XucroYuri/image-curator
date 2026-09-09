# Model and decision policy

This policy keeps model evidence separate from operational decisions. It applies to the target MoAT → NudeNet → reference open set → SigLIP → VLM → human-learning flow and to any caller-supplied adapter.

## Independent decision axes

Store and review these axes independently:

| Axis | Question | Safe default |
| --- | --- | --- |
| `safety` | Does the detector or policy indicate a safety concern? | `unknown` when the detector is unavailable, ambiguous or out of distribution. |
| `technical_quality` | Is the file technically usable for the intended purpose? | `review` when dimensions, decode, blur or corruption evidence is incomplete. |
| `identity` | Does the content match an explicitly defined reference identity or class? | `unknown` unless a calibrated open-set decision passes both similarity and margin requirements. |
| `publish_value` | Is there a reason to retain or publish this item for the stated use? | `review`; never infer from safety or identity alone. |

An item may be technically strong but unsafe, safe but low quality, identity-unknown but valuable for discovery, or suitable for private retention but unsuitable for publication. The output schema must preserve those combinations.

## Stage policy

1. Read metadata and source bytes through the read-only guard. Preserve raw metadata only when an operator has made an explicit, reviewed choice; the example configuration defaults `preserve_raw: false`.
2. Run MoAT and NudeNet only as evidence-producing stages. The built-in ONNX adapter requires explicit local model/tag paths and never downloads weights. NudeNet outputs inform `safety`; MoAT similarity supports identity routing. Neither stage is a publication decision.
3. Compare embeddings with a user-owned reference set using an open-set classifier. The classifier must expose top scores, the score margin and an `unknown` path.
4. Send only disagreement or high-value uncertainty to an independent SigLIP check. A model agreement is evidence, not ground truth.
5. If disagreement persists, a caller may provide selected reference crops to a VLM. The request must state the allowed answer set, include an abstain/unknown answer, and record the model/service version. Full-library uploads are out of scope.
6. Ask a human to resolve review items and select examples for active learning. Human labels are versioned and may be used to update references or calibrate thresholds.

## Threshold calibration

Thresholds must be calibrated against explicit human truth for the intended data distribution. Never choose a threshold from a file path, filename, prompt, model confidence alone, or an unreviewed batch. The `calibrate` command accepts explicitly labeled vectors and reports coverage, accepted accuracy, unknown false-accept rate and a confusion table.

For each release of a model, reference set or preprocessing step:

- Versioned reprocessing requires a path-free adapter manifest containing a logical version, preprocessing and postprocessing settings, and SHA-256 for every model and label file. Reference sets and human calibration labels are fingerprinted separately in each decision run. Resuming with different artifacts must fail before inference.

- maintain positive, hard-negative and open-set unknown examples;
- choose similarity and margin thresholds using a versioned calibration set;
- report coverage and false acceptance for unknowns, not only aggregate accuracy;
- review threshold changes with a human owner and retain the previous calibration;
- route samples outside the calibration envelope to `unknown` or `review`.

Thresholds are policy values, not universal facts. A stricter value may reduce coverage while improving precision; the acceptable trade-off must be chosen by the operator for the use case.

## Weights, licenses and data handling

Model weights are not distributed by this repository. Users must obtain weights independently, verify the model, dataset and service licenses, and document the selected versions. Optional Python extras install runtime libraries only; they do not download weights.

Keep source images, crops, embeddings and raw metadata out of commits and public issue reports. A remote VLM is disabled by default and requires explicit consent for the selected crop and prompt. Store only the evidence needed for audit, redact sensitive values, and define a deletion or retention period outside this package.

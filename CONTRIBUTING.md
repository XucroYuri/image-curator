# Contributing to image-curator

Thanks for helping improve `image-curator`. Small, focused pull requests are easiest to review.

## Before opening a pull request

1. Create a branch from the default branch.
2. Install the development dependencies with `python -m pip install -e ".[dev]"`.
3. Run `ruff check .` and `pytest -q`.
4. Add or update tests for behavior changes, using generated fixtures rather than personal image libraries.
5. Update documentation or the example configuration when a user-facing option changes.

Do not commit model weights, image files, credentials, access tokens, private hostnames, private network addresses, or machine-specific absolute paths.

## Pull requests

Describe the user-visible behavior, the input and output contract, and the validation command. Explain any change to read-only defaults, checkpoint compatibility, metadata retention, model stages, or the meaning of `safety`, `quality`, `publish_value`, and `unknown`.

Keep changes reviewable. Separate refactors from behavior changes, preserve backward compatibility for persisted records where practical, and document migrations when it is not practical.

## Code and data standards

The scanner should remain deterministic for the same input and configuration fingerprint. Failures should be represented in the record and checkpoint rather than silently discarded. New model integrations must expose confidence and an explicit unknown path. Any operation that can mutate a source file or move a resource requires an explicit opt-in and tests covering the refusal path.

By contributing, you agree that your contribution is provided under the Apache License 2.0 in [LICENSE](LICENSE).

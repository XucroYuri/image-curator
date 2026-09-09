# Public repository privacy boundary

This repository is a reusable software framework. It is published without a real image library, real scan output, model weights, credentials, private hostnames, private network addresses, or machine-specific absolute paths.

The examples use only synthetic, relative names such as `./sample-library`, `./curation-output/checkpoint.sqlite`, and `/path/to/model.onnx`. Replace them locally with paths that are meaningful to your environment; do not paste those local values into issues, pull requests, tests, documentation, or committed configuration.

Keep these items outside the repository:

- source images, crops, thumbnails, embeddings, feature stores, checkpoints, reports, and run logs;
- model weights, model tag files, API keys, access tokens, passwords, and service credentials;
- NAS, SMB, VPN, cloud, or workstation addresses and user names;
- prompts, EXIF/IPTC/XMP values, generated filenames, and directory names that can identify a person, project, customer, or private collection.

Tests must create synthetic fixtures under the test framework's temporary directory. They must not read from a real image library or depend on a developer's home directory. Public documentation should describe contracts and configuration shapes, not a particular deployment.

Before publishing a change, inspect the tracked file list and diff, search for machine-specific paths and credentials, and confirm that no generated artifact is staged. The `.gitignore` rules are a convenience, not a security boundary; verify the index before every public push.

The framework itself is read-only by default and stores only derived evidence when explicitly asked. Operators remain responsible for protecting local checkpoints, logs, model inputs, and any raw metadata retained outside this repository.

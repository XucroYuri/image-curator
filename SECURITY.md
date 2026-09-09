# Security policy

## Supported versions

Security fixes are developed for the latest release and the default branch. Pin a released version when deploying a pipeline and review its changelog before upgrading.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use the repository's private security advisory channel when it is enabled, or contact the maintainers through the private contact listed in the repository metadata. Include:

- the affected version or commit;
- a concise description and reproducible steps using synthetic or non-sensitive data;
- impact, including whether a read-only boundary can be bypassed;
- a suggested mitigation, if known.

Remove personal data, credentials, access tokens, private hostnames and private network addresses from reports and reproductions. The project will acknowledge a report when it is received, confirm the impact, and coordinate a disclosure timeline with the reporter.

## Safety boundaries

The default policy is read-only. Do not enable file mutation on a source library until the generated plan, checkpoint and audit log have been reviewed. Model outputs are untrusted data: treat `unknown`, low confidence and parser errors as review states, and never use them as authorization to delete or publish files.

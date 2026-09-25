# Security policy

## Supported versions

OpenQuant is pre-release (0.1.0) and nothing has been published to crates.io or PyPI.
Only the `main` branch is supported; fixes land there.

## Reporting a vulnerability

Please do not report security problems in public issues, pull requests or discussions.

Report them privately through GitHub:

1. Go to the repository's [Security tab](https://github.com/Open-Quant/openquant/security).
2. Choose **Report a vulnerability**
   ([direct link](https://github.com/Open-Quant/openquant/security/advisories/new)).
3. Describe the problem, the affected code (file, function, commit), how to reproduce
   it, and what an attacker could do with it.

Only the maintainers can see the report. We aim to acknowledge it within 7 days and to
agree a disclosure date with you once the problem is understood. Fixes are published as
a GitHub security advisory, crediting you unless you ask us not to.

## Scope

In scope: memory safety or undefined behaviour in the Rust crates, including through the
Python bindings; crashes of the Python interpreter caused by input a caller could
reasonably pass; code execution through files the library reads (CSV and other loaders);
and vulnerabilities in the build, release or CI configuration of this repository.

Out of scope: numerical results that are wrong (please open a normal bug report, they
matter a great deal to us but are not security issues), and vulnerabilities in
third-party dependencies that are already public (Dependabot tracks those; a report is
still welcome if OpenQuant is exploitable because of one).

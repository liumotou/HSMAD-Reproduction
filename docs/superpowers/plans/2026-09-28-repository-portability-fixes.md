# Repository portability completion plan

## Goal

Publish the reproducibility code that already exists in the audited server workspace, without publishing datasets, checkpoints, environments, credentials, or fabricated implementations.

## Tasks

1. Add a standard-library repository contract that detects missing method entry points and server-specific absolute roots.
2. Recover source/config/test/documentation files from `/root/autodl-tmp/HSMAD` into the isolated branch, excluding generated artifacts and large binaries.
3. Replace published runner constants that hard-code `/root/autodl-tmp/HSMAD` with repository-relative defaults plus explicit environment overrides.
4. Expand the static validator and bilingual documentation with exact commands, dependency boundaries, and honest blocked/partial statuses.
5. Run static/unit contracts locally and dependency-aware contracts plus representative smoke runs in the isolated server checkout.
6. Review the complete branch, commit it, and publish the isolated branch to GitHub.

## Constraints

- Do not modify `/root/autodl-tmp/HSMAD`.
- Do not upload datasets, results, checkpoints, caches, environments, logs, or credentials.
- Do not claim author-exact reproduction.
- Do not invent missing implementations.
- Preserve historical experiment semantics and results.


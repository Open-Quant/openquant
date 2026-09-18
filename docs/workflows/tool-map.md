# Workflow tool map

AI-DLC defines stable responsibilities and maps them to providers through
`ai-dlc.toml`. Confirm this project's selected roles; omitted capabilities are
not configured provider choices.

[Back to the workflow map](../development-workflow.md)

| Responsibility | Role or service | Default provider | Durable ownership |
| --- | --- | --- | --- |
| Formal behavior | `specs` | OpenSpec | Requirements and scenarios |
| Priority and lifecycle | `tracker` | Linear | Work identity, priority, and status |
| Personal continuity | `knowledge` | Obsidian | Private notes, reflection, and links |
| Review and merge | `scm` | GitHub | Branch, PR, merged SHA, CI runs, and artifacts |
| Deployment evidence | `deploy` | None | Environment evidence when configured |
| Interactive agent | `agent-client` | Claude Code and Codex | Analysis, authoring, and authorized tool use |
| Workflow enforcement | AI-DLC CLI and selected MCP services | Local services | CLI: validation, machine bindings, reconciliation, receipts, and gates; MCP: reviewed work, doctor, document inspection, and knowledge |
| Project lifecycle | Copier | Template service | Answers, source revision, preview, and three-way updates |

The complete publish/start/finish lifecycle requires configured tracker
and SCM roles. Local OpenSpec and GitHub compatibility fallbacks exist, but do
not select account, repository, or authorization. GitHub uses conventional
`verify.yml` and `main` defaults unless overridden; the tracker has no fallback.
Without tracker or SCM configuration, initialization, adoption, setup, checks,
design, and documentation remain available, but evidence-gated remote
completion does not. A missing specification role may deliberately use the
local OpenSpec fallback when its artifacts exist; otherwise work must be
reviewed with `requires_spec = false`. Knowledge, deployment, and agent clients
are optional.

GitHub Issues and Projects is the recommended work tracker with recorded live
evidence. Jira Cloud remains available pending its live workflow qualification;
Plane is an available, unqualified adapter outside the default toolset. Follow the
[Jira preparation record](https://github.com/Sean-Koval/ai-dlc/blob/main/docs/verification/jira-cloud-new-work.md)
for prerequisites and execution limits; adapter availability alone does not
establish readiness.

## Skills by stage

| Stage | Skill |
| --- | --- |
| Reconcile current state | `day-start` |
| Clarify an uncertain problem | `discovery` |
| Preserve product rationale | `prd-draft` |
| Decide on formal behavior | `needs-spec` |
| Translate approved behavior | `spec-from-prd` |
| Triage incoming ideas | `review-inbox` |
| Transfer verified context | `handoff` |
| Close a work period | `day-end` |

Skills provide judgment. Provider instructions define vendor operations.
AI-DLC services own validation and mutation boundaries; no skill bypasses finish
gates or extends user authorization.

## Portable enrollment boundary

A private profile repository owns the portable `ai-dlc-profile.toml` and is
enrolled by pinned revision. A second machine enrolls that same revision but
maintains its own machine binding for paths, account selection, and
environment-variable names. The project repository owns shared policy; an
external password manager, keychain, or secret injector owns credential values
and supplies them transiently through the process environment. Codex and Claude
own their generated client configuration. Never commit `.env` files or
credential values.

`ai-dlc machine status`, `plan`, `apply`, `sync`, and `doctor` own the local
enrollment lifecycle. Local CLI and MCP execution are current; hosted or cloud
execution is a later qualification target. Obsidian project portals are the default machine-local canonical file links; explicit mount mode exposes existing canonical directories from a stable checkout. Neither mode synchronizes document bodies. Provider discovery remains separately scoped.

Machine enrollment mutations are CLI-only in this cycle. MCP exposes
`work_context`, `work_publish`, `work_start`, `work_status`, `work_link`, `work_finish`,
`doctor`, `project_docs_check`, `knowledge_find`, `knowledge_append`, and `knowledge_note`,
plus the documentation and workspace tools below; it does
not expose machine enrollment mutation. These MCP identifiers differ from the
space-separated CLI commands, such as `ai-dlc work publish` and `ai-dlc
knowledge append`.

## Command groups

| Area | Interfaces |
| --- | --- |
| Readiness and context | `ai-dlc doctor`, `ai-dlc next`, `ai-dlc context` |
| Project lifecycle | `ai-dlc project init`, `ai-dlc project adopt`, `ai-dlc project sync`, `ai-dlc project setup`, `ai-dlc project check --required`, `ai-dlc project rebind` |
| Agent configuration | `ai-dlc agents render` |
| Local work drafting | `ai-dlc work new WORK_ID [--from-issue REF]` creates an unreviewed record without publication or mutation state |
| Work and traceability | `ai-dlc work publish`, `ai-dlc work link`, `ai-dlc work start`, `ai-dlc work finish` |
| Local work inspection | `ai-dlc work status` reads the local record and specification state without querying the tracker |
| Specification finalization | `ai-dlc work archive` finalizes the owned OpenSpec change, repoints the record and commits affected files before merge |
| Provider verification | `ai-dlc provider list`, `ai-dlc provider test` |
| Project documentation | `ai-dlc docs init`, `ai-dlc docs check`, `ai-dlc project link-vault` |
| Personal knowledge | `ai-dlc knowledge find`, `ai-dlc knowledge note`, `ai-dlc knowledge append` |
| Profiles and machine setup | `ai-dlc profile show`, `ai-dlc profile migrate`, `ai-dlc profile capture`; `ai-dlc setup plan`, `ai-dlc setup apply` |
| Agent-native access | `ai-dlc mcp serve` — reviewed work, read-only doctor/document inspection, and selected knowledge services; machine enrollment mutation remains CLI-only |
| Legacy compatibility | `ai-dlc scaffold` |

The local MCP server exposes only the reviewed services listed above; machine
enrollment mutation remains CLI-only. The legacy scaffold command preserves the
retired Rust-era provider interface; it is not the project lifecycle.

When replacing a tool, keep its responsibility stable where possible. Update
configuration, provider contracts, this map, relevant skills, integration
tests, live walkthrough evidence, and project templates together. Rebind
existing work explicitly; do not silently move it between providers. Portable
configuration may name environment variables, but actual secrets, account
choices, and machine-local paths remain outside portable project files.


## Selected scaffold and client tools

Preview `ai-dlc project adopt` with explicit `--tracker`, `--knowledge` and repeated
`--agent-client` options, or use the same options with `project init`. Supported
choices come from trusted definitions; selection does not log in, install tools
or create remote work. Omitted options retain historical defaults. Copier answers
preserve selections during `project sync`; authored conflicts require review.

Use `ai-dlc agents render` and its explicit apply operation to deliver the selected
native clients' owned files. Files alone do not prove client recognition or login.
Plane has an optional lifecycle adapter and guided provider connection. Select
and authenticate an explicit deployment/account/project; native guidance or
scaffolding alone does not qualify a live deployment. GitHub/Jira need no Plane installation.

Obsidian note storage uses the existing machine setting `paths.vault`. Its directory
must exist; its desktop viewer is optional, including in headless environments.
Offline readiness does not read notes or test write access. Keep personal vault
paths and contents out of shared project configuration.

## Optional Design PM route

For interface work, design-brief links the shaped outcome and RQ IDs to a draft
brief, reviewed task rubric and explicit budget. The user's existing visual tool
performs generation. Design-evaluate records candidate-specific observations,
required failures, ratings and unverified checks, with separate-session or human
review where available and explicit self-review otherwise. No extra installed
service, fixed model or mandatory paid tool is required.

Follow [design to implementation](design-to-implementation.md#optional-interface-evaluation)
for the four portable templates and original examples. Formal behavior remains
with the selected specification provider; the tracker owns delivery status, and
existing finish gates remain authoritative. The separate human calibration
protocol is unrun and does not spend the existing generic skill-evaluation budget.

## Documentation and private workspace tools

| MCP tool | CLI counterpart | Purpose |
|---|---|---|
| `project_docs_impact` | `docs review` | Find mapped documents and unmapped changes for an explicit Git comparison |
| `project_docs_disposition` | `docs review --disposition` | Emit content-bound reviewed decisions without writing evidence |
| `project_docs_record` | `docs review --disposition --evidence-id` | Merge reviewed decisions into one work item's evidence file, keeping decisions whose bound content is unchanged |
| `project_docs_gate` | `docs gate` | Check current dispositions and new objective debt |
| `project_docs_inventory` | `docs check --inventory` | Discover repository Markdown paths, exclusions and unavailable paths without reading bodies |
| `project_docs_search` | `docs search` | Search `docs/`, `openspec/` and per-call declared Markdown under one body budget; returns canonical paths, digests and explicit omissions |
| `project_docs_read` | `docs read` | Read one complete eligible project document within a byte budget; undeclared root or legacy files are refused |
| `project_docs_review` | `docs review --report` | Prepare bounded selected-document context; explicit `source=inventory` permits uncatalogued documents |
| `project_docs_review_check` | `docs review --check` | Check citations, scope and source bytes; does not certify semantic truth |
| `project_workspace_preview` | `project workspace-init` | Preview additive private project navigation; CLI apply explicitly creates files |
| `project_vault_mount_preview` | `project link-vault --mode mount --preview` | Preview canonical docs and existing OpenSpec directory mounts; omit CLI `--preview` to apply |
| CLI only | `project workspace-init --shell` | Preview the owned bash, zsh or fish PATH section; `--apply` writes after ownership checks |
| `project_workspace_check` | `project workspace-check` | Report installation, shell activation, each mount binding and link navigation separately; read-only, native client not assessed |

Project-document search and read cover repository files, never private notes.
Declare additional Markdown per call with repeated `--source` (MCP `sources`);
nothing is persisted. Edit returned repository paths with ordinary file and Git
tools and run project checks. The `knowledge_*` tools remain for private notes,
and a vault portal, mounted folder or Markdown link does not grant access.

The [documentation guide](../documentation-guide.md#search-and-read-project-documents) explains ownership, packet
review, project-document access, baselines and workspace use. The optional CLI `docs check --style` invokes
configured Vale; it does not install a tool or establish factual correctness.

### Session continuity

`ai-dlc work finish <id> --learning FILE` optionally stores an authored learning
through the knowledge provider after completion gates pass. MCP `work_finish`
accepts `learning` text alongside its existing `handoff`. Missing knowledge leaves
the note pending and preserves completion. `work start` returns up to five
matching learning paths and first lines; read relevant notes before implementing.
The session-start hook recalls notes for the branch's bound work record. Stop
reminders use only ignored local friction counts and never transmit them.

### Local frontend evidence

The optional node `frontend` capability supplies a Playwright smoke check.
`ai-dlc design capture` writes viewport screenshots and a manifest for
`design-evaluate`; browser installation belongs to explicit setup. A named state
waits for its visible selector and does not certify untested interactions.
See [frontend smoke and capture](design-to-implementation.md#frontend-smoke-and-capture-evidence).

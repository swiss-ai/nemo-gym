# Workspace SWE

Interactive coding tasks with the existing `gymnasium_agent`: use `<cmd>...</cmd>`
to run a shell command, `<write path="relative/path">...</write>` to replace a
file, and `<submit/>` to grade. The example task repairs `mathutils.py`.

`configs/workspace_swe.yaml` launches the environment locally.
`configs/workspace_swe_remote.yaml` connects the local agent to the deployed
`https://gym-dev.swissai.svc.cscs.ch/workspace_swe` service. Remote configurations
do not install or configure that service. Its deployment must provide Python,
pytest, and git when tasks clone repositories.

The remote config opts into `legacy_responses_compatibility`: outgoing resource
requests omit optional null fields from the Responses request/response envelopes
and encode unknown usage details as zero for verification. Original trainer
responses, non-null options, and nested input/tool/task data are preserved.
Remove this opt-in once the hosted service accepts the current Responses schemas
and nullable usage details.

Dataset extras supply `task`, `files`, `hidden_tests`, `test_cmd`, and `max_steps`.
Optional `repo_url`, `commit`, and `setup_cmd` support repository tasks. Hidden
tests are written at submission or the step limit. Reward is the pytest summary's
pass fraction; test commands must report pytest-style summaries. This verifier
does not protect grading from adversarial code running in the same interpreter
environment.

Each session receives a temporary directory under `workspace_dir` (default
`/workspaces`), removed on completion, explicit close, reset, cancellation, or
expiry at a subsequent reset. Reset advertises `supports_explicit_close`, so the
Gymnasium agent also closes the workspace when a model call fails or its own
step limit ends the rollout.
Commands execute asynchronously with bounded concurrency and timeouts. They
execute arbitrary code with the service's permissions: directories do **not**
isolate sessions, the host filesystem, or the network. Run this service in an
appropriately isolated execution environment without credentials. A replicated
deployment must preserve session affinity because workspace state is local to
each service process.

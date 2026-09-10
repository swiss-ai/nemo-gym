# Workspace SWE

Interactive coding tasks with the existing `gymnasium_agent`: use `<cmd>...</cmd>`
to run a shell command, `<write path="relative/path">...</write>` to replace a
file, and `<submit/>` to grade. The example task repairs `mathutils.py`.

`configs/workspace_swe.yaml` launches the environment locally.
`configs/workspace_swe_remote.yaml` connects the local agent to the deployed
`https://gym-dev.swissai.svc.cscs.ch/workspace_swe` service. Remote configurations
do not install or configure that service. Its deployment must provide Python,
pytest, and git when tasks clone repositories.

Dataset extras supply `task`, `files`, `hidden_tests`, `test_cmd`, and `max_steps`.
Optional `repo_url`, `commit`, and `setup_cmd` support repository tasks. Hidden
tests are written at submission or the step limit. Reward is the pytest summary's
pass fraction; test commands must report pytest-style summaries. This verifier
does not protect grading from adversarial code running in the same interpreter
environment.

Each session receives a temporary directory under `workspace_dir` (default
`/workspaces`), removed on completion, reset, or expiry at a subsequent reset.
Commands execute asynchronously with bounded concurrency and timeouts. They
execute arbitrary code with the service's permissions: directories do **not**
isolate sessions, the host filesystem, or the network. Run this service in an
appropriately isolated execution environment without credentials. A replicated
deployment must preserve session affinity because workspace state is local to
each service process.

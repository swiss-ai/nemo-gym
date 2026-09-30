"""Run Harbor task sandboxes as pods on an ordinary Kubernetes cluster.

Harbor 0.1.42 ships one Kubernetes-backed environment, `harbor.environments.gke`,
and it cannot be pointed at a cluster that is not GKE:

  - it ignores `[environment].docker_image` and synthesises a Google Artifact
    Registry URL instead (gke.py:320-322);
  - it shells out to `gcloud` for cluster credentials, image existence and Cloud
    Build (gke.py:69-90, 324-337, 358-381);
  - its pod spec carries no imagePullSecrets, service account, node selector or
    tolerations (gke.py:429-456), so a private registry and tainted or
    heterogeneous nodes are both out of reach.

This class keeps what is generic in that implementation -- pod create/delete,
command execution over the API server's exec endpoint, file transfer as tar
streamed through the same endpoint -- and replaces the Google-specific parts:
the image comes from task.toml, and the pod spec exposes the knobs a shared
multi-tenant cluster actually needs.

Selected the same way as the Apptainer environment next door, by import path
rather than by `EnvironmentType`, since the enum has no Kubernetes member:

    harbor_environment_import_path: "responses_api_agents.harbor_agent.custom_envs.k8s.environment:KubernetesEnvironment"
    harbor_environment_kwargs:
      namespace: swiss-ai-sandbox
      image_pull_secrets: [ghcr-harbor-pull]

Two differences from the Apptainer sibling are worth stating, because they are
what make this path cheaper rather than merely different:

  1. No in-container server. `SingularityEnvironment` starts uvicorn/fastapi
     inside the sandbox and talks to it over HTTP, which is where its
     "Failed to start Singularity FastAPI server" timeouts come from. Here the
     API server's exec endpoint already provides that channel, so there is no
     server to install, start or wait for.
  2. No bind mounts. Apptainer mounts the task's `environment/files/` read-only
     at `$HARBOR_STAGING`; a pod cannot see the submitting node's filesystem, so
     the same directory is uploaded instead. The contract the task sees --
     `$WORKDIR`, `$HARBOR_STAGING`, and a `setup.sh` run from the latter -- is
     reproduced exactly, so a task written for either environment runs on both.

Because the pod is not mounted, `is_mounted` is False and Harbor downloads the
agent and verifier log directories over the same exec channel
(trial.py:318, verifier.py:141).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
import shlex
import tarfile
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream
from tenacity import retry, stop_after_attempt, wait_exponential


# Where the task's environment/files/ is uploaded. Same path the Apptainer
# environment bind-mounts it at, so a setup.sh cannot tell the two apart.
HARBOR_STAGING = "/staging/env_files"

# Fallback container workdir when the Dockerfile declares no WORKDIR. Matches
# SingularityEnvironment._resolve_workdir and Harbor's task convention.
DEFAULT_WORKDIR = "/app"

# A pod name is a DNS-1123 label.
MAX_POD_NAME_LEN = 63

POD_READY_TIMEOUT_SEC = 300
EXEC_READY_ATTEMPTS = 60
EXEC_READY_INTERVAL_SEC = 3
POD_DELETE_TIMEOUT_SEC = 60


class _SharedClients:
    """Process-wide cache of Kubernetes API clients, keyed by kubeconfig+context.

    A NeMo-RL step runs many rollouts concurrently in one process, and each
    builds its own environment. Handing them a shared client keeps one
    connection pool to the API server instead of one per sandbox.

    Each entry gets its own `Configuration` rather than going through
    `load_kube_config`'s default, which mutates global state and would make two
    differently-configured environments in the same process interfere.
    """

    _lock = asyncio.Lock()
    _clients: dict[tuple[str | None, str | None], k8s_client.CoreV1Api] = {}

    @classmethod
    async def get(cls, kubeconfig: str | None, context: str | None) -> k8s_client.CoreV1Api:
        """Return the shared CoreV1Api for one kubeconfig/context pair.

        Args:
            kubeconfig: Path to a kubeconfig file, or None for the default
                lookup (KUBECONFIG, then ~/.kube/config, then in-cluster).
            context: Context name within that file, or None for its current one.

        Returns:
            A CoreV1Api bound to a dedicated ApiClient.
        """
        key = (kubeconfig, context)
        async with cls._lock:
            api = cls._clients.get(key)
            if api is None:
                api = await asyncio.to_thread(cls._build, kubeconfig, context)
                cls._clients[key] = api
            return api

    @staticmethod
    def _build(kubeconfig: str | None, context: str | None) -> k8s_client.CoreV1Api:
        """Load credentials and construct an isolated API client.

        Args:
            kubeconfig: Path to a kubeconfig file, or None.
            context: Context name, or None.

        Returns:
            A CoreV1Api with its own Configuration and ApiClient.

        Raises:
            RuntimeError: If no usable credentials were found.
        """
        configuration = k8s_client.Configuration()
        try:
            k8s_config.load_kube_config(
                config_file=kubeconfig,
                context=context,
                client_configuration=configuration,
            )
        except Exception as kubeconfig_error:
            # A pod that carries a service account can reach the API server
            # without a kubeconfig at all; try that before giving up.
            try:
                k8s_config.load_incluster_config(client_configuration=configuration)
            except Exception:
                raise RuntimeError(
                    "Could not load Kubernetes credentials. Point kubeconfig= at a "
                    "file readable from wherever the rollout runs -- on Slurm that "
                    "means a shared filesystem, not the login node's home -- or run "
                    f"inside a pod with a service account. ({kubeconfig_error})"
                ) from kubeconfig_error
        return k8s_client.CoreV1Api(k8s_client.ApiClient(configuration))


class KubernetesEnvironment(BaseEnvironment):
    """A Harbor sandbox backed by a single Kubernetes pod."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        namespace: str = "default",
        kubeconfig: str | None = None,
        context: str | None = None,
        image_pull_secrets: list[str] | None = None,
        service_account_name: str | None = None,
        node_selector: dict[str, str] | None = None,
        tolerations: list[dict] | None = None,
        memory_limit_multiplier: float | None = None,
        pod_name_prefix: str = "harbor",
        pod_ready_timeout_sec: int = POD_READY_TIMEOUT_SEC,
        workdir: str | None = None,
        *args,
        **kwargs,
    ):
        """Configure the sandbox.

        Args:
            environment_dir: Task's `environment/` directory.
            environment_name: Task name.
            session_id: Trial name, `<task_name>__<trial_id>`.
            trial_paths: Where the trial writes its output.
            task_env_config: The task's `[environment]` table.
            namespace: Namespace the pod is created in.
            kubeconfig: Kubeconfig path; None uses the default lookup.
            context: Kubeconfig context; None uses the file's current one.
            image_pull_secrets: Secret names granting access to the image's
                registry. Required for a private GHCR package.
            service_account_name: Service account for the pod.
            node_selector: Node labels the pod must match.
            tolerations: Taints the pod tolerates, as V1Toleration mappings.
                Without these, tainted nodes are unusable, which on a small
                cluster can be most of it.
            memory_limit_multiplier: Memory limit as a multiple of the task's
                request. None leaves the pod burstable, which is what the task
                configs assume.
            pod_name_prefix: Prefix for generated pod names, so a shared
                namespace can be filtered by owner.
            pod_ready_timeout_sec: How long to wait for the pod to become Ready.
                Worth raising when the image is large and uncached, since the
                first pod on each node pays the pull.
            workdir: Container working directory. Defaults to the Dockerfile's
                WORKDIR, then /app.
        """
        self._namespace = namespace
        self._kubeconfig = kubeconfig
        self._context = context
        self._image_pull_secrets = list(image_pull_secrets or [])
        self._service_account_name = service_account_name
        self._node_selector = dict(node_selector or {})
        self._tolerations = list(tolerations or [])
        self._memory_limit_multiplier = memory_limit_multiplier
        self._pod_name_prefix = pod_name_prefix
        self._pod_ready_timeout_sec = pod_ready_timeout_sec
        self._workdir_override = workdir

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            *args,
            **kwargs,
        )

        self._core_api: k8s_client.CoreV1Api | None = None
        self._pod_name = self._build_pod_name()
        self._workdir = self._resolve_workdir()

    @staticmethod
    def type() -> EnvironmentType:
        """The closest enum member; Harbor's EnvironmentType has no Kubernetes value.

        Returns:
            EnvironmentType.GKE.
        """
        # Only ever read for its .value, in the GPU and internet validation
        # messages in BaseEnvironment. Returning a real member matters: the
        # Apptainer sibling returns EnvironmentType.SINGULARITY, which does not
        # exist, so those two error paths raise AttributeError instead of the
        # message they were meant to produce.
        return EnvironmentType.GKE

    @property
    def is_mounted(self) -> bool:
        """Whether Harbor's log directories are visible without downloading them.

        Returns:
            False; a pod cannot mount the submitting node's filesystem.
        """
        return False

    @property
    def supports_gpus(self) -> bool:
        """Whether the sandbox can be given GPUs.

        Returns:
            False. The cluster's GPUs belong to training, and Harbor terminal
            tasks are CPU work.
        """
        return False

    @property
    def can_disable_internet(self) -> bool:
        """Whether `allow_internet = false` can be honoured.

        Returns:
            False; isolating a pod needs a NetworkPolicy this class does not
            create, and silently ignoring the request would be worse.
        """
        return False

    def _validate_definition(self) -> None:
        """Check the task names an image this environment can pull.

        Raises:
            ValueError: If `docker_image` is missing, or names a `.sif`, which
                belongs to the Apptainer environment.
        """
        image = self.task_env_config.docker_image
        if not image:
            raise ValueError(
                "The Kubernetes environment needs 'docker_image' in task.toml "
                "[environment]. Set it to a registry reference with "
                "prepare_harbor_dataset.py --image <ref>."
            )
        if image.endswith(".sif"):
            raise ValueError(
                f"docker_image is an Apptainer image ({image}), which Kubernetes "
                "cannot run. Build a registry image with "
                "examples/nemo_gym/build_harbor_task_image_kubernetes.py and point "
                "the task tree at it."
            )

    def _build_pod_name(self) -> str:
        """Derive a DNS-1123 pod name from the session id.

        Truncation alone is unsafe here: Harbor session ids are
        `<task_name>__<trial_id>` and the task names in a subset share a long
        prefix, so cutting to 63 characters can map two live trials onto one
        pod. Anything overlong therefore keeps a digest of the full id.

        Returns:
            A valid, collision-resistant pod name.
        """
        raw = f"{self._pod_name_prefix}-{self.session_id}".lower()
        name = re.sub(r"[^a-z0-9-]+", "-", raw).strip("-")
        if len(name) > MAX_POD_NAME_LEN:
            digest = hashlib.sha256(self.session_id.encode("utf-8")).hexdigest()[:10]
            name = f"{name[: MAX_POD_NAME_LEN - len(digest) - 1]}-{digest}"
        return name.strip("-")

    def _resolve_workdir(self) -> str:
        """Resolve the container working directory: kwarg, then Dockerfile, then default.

        The Dockerfile is consulted because that is where the image's own
        WORKDIR is declared, and the task's instructions are written against it.

        Returns:
            An absolute path inside the container.
        """
        if self._workdir_override and self._workdir_override.strip():
            return self._workdir_override.strip()

        dockerfile = self.environment_dir / "Dockerfile"
        if dockerfile.is_file():
            workdir = None
            for line in dockerfile.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip().upper().startswith("WORKDIR "):
                    # Last one wins, as in Docker.
                    workdir = line.strip()[len("WORKDIR ") :].strip().strip("\"'")
            if workdir:
                return workdir
        return DEFAULT_WORKDIR

    @property
    def pod_name(self) -> str:
        """The name of this sandbox's pod.

        Returns:
            The generated pod name.
        """
        return self._pod_name

    async def _ensure_client(self) -> None:
        """Bind this environment to the shared API client."""
        if self._core_api is None:
            self._core_api = await _SharedClients.get(self._kubeconfig, self._context)

    def _pod_manifest(self) -> k8s_client.V1Pod:
        """Build the pod that backs this sandbox.

        Returns:
            The pod object to create.
        """
        requests = {
            "cpu": str(self.task_env_config.cpus),
            # Mi directly, rather than dividing into Gi, to avoid rounding a
            # task's request down to something it cannot run in.
            "memory": f"{self.task_env_config.memory_mb}Mi",
            "ephemeral-storage": f"{self.task_env_config.storage_mb}Mi",
        }
        limits = {}
        if self._memory_limit_multiplier:
            limit_mb = int(self.task_env_config.memory_mb * self._memory_limit_multiplier)
            limits["memory"] = f"{limit_mb}Mi"

        return k8s_client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=k8s_client.V1ObjectMeta(
                name=self._pod_name,
                namespace=self._namespace,
                labels={
                    "app": "harbor-sandbox",
                    # Truncated because a label value is limited to 63
                    # characters; these are for filtering, not identity.
                    "harbor-task": re.sub(r"[^A-Za-z0-9_.-]+", "-", self.environment_name)[:63],
                },
            ),
            spec=k8s_client.V1PodSpec(
                containers=[
                    k8s_client.V1Container(
                        name="main",
                        image=self.task_env_config.docker_image,
                        # The sandbox is driven entirely through exec, so the
                        # container only has to stay alive.
                        command=["sleep", "infinity"],
                        working_dir=self._workdir,
                        resources=k8s_client.V1ResourceRequirements(requests=requests, limits=limits or None),
                    )
                ],
                image_pull_secrets=[
                    k8s_client.V1LocalObjectReference(name=secret) for secret in self._image_pull_secrets
                ]
                or None,
                service_account_name=self._service_account_name,
                node_selector=self._node_selector or None,
                tolerations=[k8s_client.V1Toleration(**toleration) for toleration in self._tolerations] or None,
                # Never: a sandbox that died is a failed trial, and restarting it
                # would silently discard the agent's work so far.
                restart_policy="Never",
            ),
        )

    async def start(self, force_build: bool) -> None:
        """Create the pod and prepare the task's working directory.

        Args:
            force_build: Ignored. The image is built ahead of time by
                build_harbor_task_image_kubernetes.py; there is no Docker daemon
                here to build one on demand, and building per trial would repeat
                the same work once per rollout.

        Raises:
            RuntimeError: If the pod cannot be created or does not become ready.
        """
        del force_build

        await self._ensure_client()
        await self._create_pod()
        await self._wait_for_pod_ready()
        await self._wait_for_exec_ready()

        result = await self.exec(
            f"mkdir -p {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir} {shlex.quote(self._workdir)}"
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"Could not create the log directories in pod {self._pod_name}: {result.stderr or result.stdout}"
            )

        await self._stage_environment_files()

    async def _create_pod(self) -> None:
        """Create the pod, replacing one left behind by an earlier trial.

        Raises:
            RuntimeError: If creation fails, or a stale pod cannot be removed.
        """
        pod = self._pod_manifest()
        try:
            await asyncio.to_thread(
                self._core_api.create_namespaced_pod,
                namespace=self._namespace,
                body=pod,
            )
            return
        except ApiException as error:
            if error.status != 409:
                raise RuntimeError(f"Could not create pod {self._pod_name}: {error.status} {error.reason}") from error

        # 409: a pod of this name survived a previous run, so its filesystem
        # holds another trial's state. Replace it rather than adopt it.
        self.logger.debug(f"Pod {self._pod_name} already exists; replacing it.")
        await self._delete_pod(wait=True)
        try:
            await asyncio.to_thread(
                self._core_api.create_namespaced_pod,
                namespace=self._namespace,
                body=pod,
            )
        except ApiException as error:
            raise RuntimeError(f"Could not re-create pod {self._pod_name}: {error.status} {error.reason}") from error

    async def _stage_environment_files(self) -> None:
        """Put the task's `environment/files/` into the sandbox.

        Reproduces both halves of what the task expects:

          - `COPY files/ /app/` from the task Dockerfile, which the shared image
            cannot carry because one image serves a whole subset while the files
            differ per task;
          - the `setup.sh` hook, run with `$WORKDIR` and `$HARBOR_STAGING` set
            exactly as SingularityEnvironment's bootstrap sets them.

        The copies are the agent's to edit. `setup.sh` itself is not copied: it
        is plumbing, not a task input, and Docker's `COPY` would have placed it
        in the workdir only as a side effect.

        Raises:
            RuntimeError: If staging fails.
        """
        files_dir = self.environment_dir / "files"
        if not files_dir.is_dir():
            return

        await self.upload_dir(files_dir, HARBOR_STAGING)

        script = (
            "set -e\n"
            f"export WORKDIR={shlex.quote(self._workdir)}\n"
            f"export HARBOR_STAGING={shlex.quote(HARBOR_STAGING)}\n"
            'mkdir -p "$WORKDIR"\n'
            'for entry in "$HARBOR_STAGING"/* "$HARBOR_STAGING"/.[!.]*; do\n'
            '  [ -e "$entry" ] || continue\n'
            '  if [ "$(basename "$entry")" = setup.sh ]; then continue; fi\n'
            '  cp -a "$entry" "$WORKDIR"/\n'
            "done\n"
            'if [ -f "$HARBOR_STAGING/setup.sh" ]; then\n'
            '  bash "$HARBOR_STAGING/setup.sh"\n'
            "fi\n"
        )
        result = await self.exec(script)
        if result.return_code != 0:
            raise RuntimeError(
                f"Staging environment/files/ into {self._workdir} failed in pod "
                f"{self._pod_name}: {result.stderr or result.stdout}"
            )

    async def stop(self, delete: bool) -> None:
        """Tear the sandbox down.

        Args:
            delete: Whether to delete the pod. False keeps it for inspection,
                which also keeps it billing CPU and memory on a shared cluster.
        """
        if self._core_api is None or not delete:
            return
        try:
            await self._delete_pod(wait=False)
        except ApiException as error:
            if error.status != 404:
                self.logger.warning(f"Could not delete pod {self._pod_name}: {error}")

    async def _delete_pod(self, wait: bool) -> None:
        """Delete the pod, optionally blocking until it is gone.

        Args:
            wait: Whether to wait for the API server to stop reporting the pod.
                Needed before re-creating it under the same name.

        Raises:
            RuntimeError: If `wait` is set and the pod outlives the timeout.
        """
        try:
            await asyncio.to_thread(
                self._core_api.delete_namespaced_pod,
                name=self._pod_name,
                namespace=self._namespace,
                body=k8s_client.V1DeleteOptions(grace_period_seconds=0, propagation_policy="Foreground"),
            )
        except ApiException as error:
            if error.status == 404:
                return
            raise

        if not wait:
            return

        for _ in range(POD_DELETE_TIMEOUT_SEC):
            try:
                await asyncio.to_thread(
                    self._core_api.read_namespaced_pod,
                    name=self._pod_name,
                    namespace=self._namespace,
                )
            except ApiException as error:
                if error.status == 404:
                    return
                raise
            await asyncio.sleep(1)
        raise RuntimeError(
            f"Pod {self._pod_name} was still present {POD_DELETE_TIMEOUT_SEC}s after deletion was requested."
        )

    async def _wait_for_pod_ready(self) -> None:
        """Block until the pod is Running with all containers ready.

        Raises:
            RuntimeError: If the pod fails, cannot pull its image, or does not
                become ready in time.
        """
        for elapsed in range(self._pod_ready_timeout_sec):
            try:
                pod = await asyncio.to_thread(
                    self._core_api.read_namespaced_pod,
                    name=self._pod_name,
                    namespace=self._namespace,
                )
            except ApiException as error:
                if error.status != 404:
                    raise RuntimeError(
                        f"Kubernetes API error while waiting for {self._pod_name}: {error.status} {error.reason}"
                    ) from error
                await asyncio.sleep(1)
                continue

            phase = pod.status.phase
            statuses = pod.status.container_statuses or []
            if phase == "Running" and statuses and all(c.ready for c in statuses):
                return
            if phase in ("Failed", "Unknown"):
                raise RuntimeError(f"Pod {self._pod_name} failed to start: {self._failure_summary(pod)}")
            # A bad image reference or a missing pull secret never resolves, so
            # report it now instead of after the full timeout.
            for status in statuses:
                waiting = status.state.waiting if status.state else None
                if waiting and waiting.reason in (
                    "ImagePullBackOff",
                    "ErrImagePull",
                    "InvalidImageName",
                ):
                    raise RuntimeError(
                        f"Pod {self._pod_name} cannot pull "
                        f"{self.task_env_config.docker_image}: "
                        f"{waiting.message or waiting.reason}. If the registry is "
                        "private, image_pull_secrets must name a secret in "
                        f"namespace {self._namespace}."
                    )

            if elapsed and elapsed % 30 == 0:
                self.logger.debug(f"Pod {self._pod_name} still {phase} after {elapsed}s.")
            await asyncio.sleep(1)

        raise RuntimeError(
            f"Pod {self._pod_name} was not ready within "
            f"{self._pod_ready_timeout_sec}s. On a busy cluster this is usually "
            "scheduling or an uncached image pull rather than a broken task."
        )

    def _failure_summary(self, pod: k8s_client.V1Pod) -> str:
        """Summarise why a pod is not running.

        Args:
            pod: The pod as last read from the API server.

        Returns:
            A one-line description, or "unknown error".
        """
        reasons = []
        if pod.status.reason:
            reasons.append(pod.status.reason)
        if pod.status.message:
            reasons.append(pod.status.message)
        for status in pod.status.container_statuses or []:
            state = status.state
            if state and state.waiting:
                reasons.append(f"{status.name} waiting: {state.waiting.reason}")
            elif state and state.terminated:
                reasons.append(
                    f"{status.name} terminated: {state.terminated.reason} (exit {state.terminated.exit_code})"
                )
        return "; ".join(reasons) if reasons else "unknown error"

    async def _wait_for_exec_ready(self) -> None:
        """Block until the container accepts exec requests.

        A pod can report Ready a moment before its container is attachable, and
        the first exec then fails with a 500 that looks like a broken sandbox.

        Raises:
            RuntimeError: If exec never becomes available.
        """
        last_error: Exception | None = None
        for _ in range(EXEC_READY_ATTEMPTS):
            try:
                response = await asyncio.to_thread(
                    stream,
                    self._core_api.connect_get_namespaced_pod_exec,
                    self._pod_name,
                    self._namespace,
                    command=["true"],
                    stderr=False,
                    stdin=False,
                    stdout=True,
                    tty=False,
                    _preload_content=False,
                )
                response.close()
                return
            except Exception as error:  # noqa: BLE001 - any failure means retry
                last_error = error
                await asyncio.sleep(EXEC_READY_INTERVAL_SEC)
        raise RuntimeError(
            f"Pod {self._pod_name} did not accept exec after "
            f"{EXEC_READY_ATTEMPTS * EXEC_READY_INTERVAL_SEC}s: {last_error}"
        )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        """Run a shell command inside the sandbox.

        Args:
            command: Command to run.
            cwd: Working directory; None uses the container's own.
            env: Extra environment variables.
            timeout_sec: Seconds to wait before giving up.

        Returns:
            The command's stdout, stderr and exit status.

        Raises:
            TimeoutError: If `timeout_sec` elapses first.
        """
        await self._ensure_client()

        # -i, matching Harbor's Docker and GKE environments, so the sandbox's
        # interactive shell profile is sourced and tools installed into a
        # non-default prefix are on PATH.
        full_command = f"bash -ic {shlex.quote(command)}"
        for key, value in (env or {}).items():
            full_command = f"{key}={shlex.quote(value)} {full_command}"
        if cwd:
            full_command = f"cd {shlex.quote(cwd)} && {full_command}"

        response = None
        try:
            response = await asyncio.to_thread(
                stream,
                self._core_api.connect_get_namespaced_pod_exec,
                self._pod_name,
                self._namespace,
                command=["sh", "-c", full_command],
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False,
            )
            reader = asyncio.to_thread(self._drain, response)
            if timeout_sec:
                stdout, stderr = await asyncio.wait_for(reader, timeout=timeout_sec)
            else:
                stdout, stderr = await reader

            response.run_forever(timeout=0)
            return ExecResult(
                stdout=stdout,
                stderr=stderr,
                return_code=response.returncode if response.returncode is not None else 0,
            )
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:  # noqa: BLE001 - closing must not mask the result
                    pass

    @staticmethod
    def _drain(response) -> tuple[str, str]:
        """Read an exec stream to completion.

        Args:
            response: An open WSClient.

        Returns:
            Its stdout and stderr.
        """
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        while response.is_open():
            response.update(timeout=1)
            if response.peek_stdout():
                stdout_chunks.append(response.read_stdout())
            if response.peek_stderr():
                stderr_chunks.append(response.read_stderr())
        return "".join(stdout_chunks), "".join(stderr_chunks)

    async def _extract_tar(self, tar_bytes: bytes, target_dir: str) -> None:
        """Unpack a tar archive inside the sandbox.

        The exec endpoint is the only channel into the pod, so `kubectl cp`'s
        approach is used directly: pipe a tar into `tar xf -`.

        Args:
            tar_bytes: The archive.
            target_dir: Directory to extract into; created if absent.

        Raises:
            RuntimeError: If the directory cannot be created.
        """
        result = await self.exec(f"mkdir -p {shlex.quote(target_dir)}")
        if result.return_code != 0:
            raise RuntimeError(
                f"Could not create {target_dir} in pod {self._pod_name}: {result.stderr or result.stdout}"
            )

        response = await asyncio.to_thread(
            stream,
            self._core_api.connect_get_namespaced_pod_exec,
            self._pod_name,
            self._namespace,
            command=["tar", "xf", "-", "-C", target_dir],
            stderr=True,
            stdin=True,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        try:
            response.write_stdin(tar_bytes)
            response.run_forever(timeout=1)
        finally:
            response.close()

    async def _read_tar(self, command: list[str]) -> bytes:
        """Run a tar-producing command in the sandbox and collect its output.

        Args:
            command: Argv whose stdout is a tar archive.

        Returns:
            The archive bytes.

        Raises:
            RuntimeError: If the command produced nothing.
        """
        response = await asyncio.to_thread(
            stream,
            self._core_api.connect_get_namespaced_pod_exec,
            self._pod_name,
            self._namespace,
            command=command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        chunks: list[bytes] = []
        stderr = ""
        try:
            while response.is_open():
                response.update(timeout=1)
                if response.peek_stdout():
                    data = response.read_stdout()
                    # The stream yields str; tar output is binary, so the
                    # surrogateescape round-trip is what preserves it.
                    chunks.append(data.encode("utf-8", errors="surrogateescape") if isinstance(data, str) else data)
                if response.peek_stderr():
                    stderr += response.read_stderr()
        finally:
            response.close()

        if not chunks:
            raise RuntimeError(
                f"No data came back from {' '.join(command)} in pod {self._pod_name}: {stderr.strip() or 'no stderr'}"
            )
        return b"".join(chunks)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Copy a local file into the sandbox.

        Args:
            source_path: Local file.
            target_path: Destination path inside the container.
        """
        await self._ensure_client()
        source_path = Path(source_path)

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            tar.add(str(source_path), arcname=Path(target_path).name)
        await self._extract_tar(buffer.getvalue(), str(Path(target_path).parent))

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Copy a local directory into the sandbox.

        Args:
            source_dir: Local directory.
            target_dir: Destination directory inside the container.
        """
        await self._ensure_client()
        source_dir = Path(source_dir)

        buffer = io.BytesIO()
        count = 0
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for item in sorted(source_dir.rglob("*")):
                # Directories are carried too, so an empty one the task expects
                # to exist still arrives.
                if item.is_file() or item.is_dir():
                    tar.add(
                        str(item),
                        arcname=str(item.relative_to(source_dir)),
                        recursive=False,
                    )
                    count += 1
        if count == 0:
            self.logger.debug(f"Nothing to upload from {source_dir}")
            return

        await self._extract_tar(buffer.getvalue(), target_dir)
        self.logger.debug(f"Uploaded {count} entries to {target_dir}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Copy a file out of the sandbox.

        Args:
            source_path: File inside the container.
            target_path: Local destination.

        Raises:
            RuntimeError: If the archive holds no regular file.
        """
        await self._ensure_client()
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        tar_bytes = await self._read_tar(["tar", "cf", "-", source_path])
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                target_path.write_bytes(extracted.read())
                return
        raise RuntimeError(f"{source_path} in pod {self._pod_name} is not a regular file.")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Copy a directory out of the sandbox.

        This is how the trial's agent and verifier logs come back, since the pod
        mounts nothing (`is_mounted` is False).

        Args:
            source_dir: Directory inside the container.
            target_dir: Local destination; existing files are overwritten.
        """
        await self._ensure_client()
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        tar_bytes = await self._read_tar(["sh", "-c", f"cd {shlex.quote(source_dir)} && tar cf - ."])
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            # "data" rejects absolute paths, parent traversal and special files.
            # The sandbox ran agent-authored code, so its output is untrusted.
            tar.extractall(path=str(target_dir), filter="data")

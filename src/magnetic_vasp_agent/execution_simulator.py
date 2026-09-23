"""Local fixed-program lifecycle exercise; never executes user commands or VASP."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

from magnetic_vasp_agent.execution import write_report

Scenario = Literal["complete", "fail", "timeout", "cancel"]
PROGRAMS = {
    "complete": "print('SIMULATION ONLY: completed without scientific results', flush=True)",
    "fail": "import sys; print('SIMULATION ONLY: intentional failure', file=sys.stderr, flush=True); sys.exit(7)",
    "timeout": "import time; print('SIMULATION ONLY: waiting', flush=True); time.sleep(30)",
    "cancel": "import time; print('SIMULATION ONLY: waiting', flush=True); time.sleep(30)",
}


class LocalSimulation:
    """Only a held Popen handle can be cancelled; accepting arbitrary PIDs is forbidden."""

    def __init__(self, output: Path, scenario: Scenario, timeout_seconds: float = 2):
        if scenario not in PROGRAMS:
            raise ValueError("unsupported simulation scenario")
        if not 0.05 <= timeout_seconds <= 10:
            raise ValueError("simulation timeout must be between 0.05 and 10 seconds")
        self.output = output.absolute()
        self.scenario = scenario
        self.timeout_seconds = timeout_seconds
        self.process: subprocess.Popen | None = None
        self.state = "prepared"
        self.returncode: int | None = None
        self.events = [self.state]
        self._stdout = self._stderr = None
        self._started = 0.0

    def start(self) -> None:
        if self.state != "prepared":
            raise ValueError("simulation can only be started once")
        self.output.mkdir(parents=True, exist_ok=False)
        try:
            self._stdout = (self.output / "stdout.log").open("xb")
            self._stderr = (self.output / "stderr.log").open("xb")
            self.process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-u", "-c", PROGRAMS[self.scenario]],
                cwd=self.output, stdin=subprocess.DEVNULL,
                stdout=self._stdout, stderr=self._stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._started = time.monotonic()
            self.state = "running"
            self.events.append(self.state)
        except BaseException:
            self._finish("start_failed")
            raise

    def _finish(self, state: str) -> None:
        self.state = state
        self.events.append(state)
        for handle in (self._stdout, self._stderr):
            if handle is not None:
                handle.close()

    def poll(self) -> str:
        if self.state == "running":
            self.returncode = self.process.poll()
            if self.returncode is not None:
                self._finish("completed_unverified" if self.returncode == 0 else "failed")
            elif time.monotonic() - self._started >= self.timeout_seconds:
                self._stop("timed_out")
        return self.state

    def _stop(self, state: str) -> None:
        # The fixed child does not spawn descendants. Remote MPI needs a different ownership design.
        self.process.terminate()
        try:
            self.returncode = self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.returncode = self.process.wait(timeout=2)
        self._finish(state)

    def cancel(self) -> None:
        if self.poll() == "running":
            self._stop("cancelled")

    def close(self) -> None:
        self.cancel()

    def report(self) -> dict:
        self.poll()
        return {
            "schema_version": 1, "mode": "local_simulation", "scenario": self.scenario,
            "state": self.state, "exit_code": self.returncode, "events": list(self.events),
            "scientific_converged": None, "runtime_verified": False,
            "artifacts_released": False, "retry_count": 0,
            "stdout": str(self.output / "stdout.log"),
            "stderr": str(self.output / "stderr.log"),
        }


def simulate(output: Path, scenario: Scenario) -> dict:
    simulation = LocalSimulation(output, scenario)
    try:
        simulation.start()
        if scenario == "cancel":
            simulation.cancel()
        while simulation.poll() == "running":
            time.sleep(0.02)
        report = simulation.report()
        write_report(report, simulation.output / "simulation.manifest.json")
        return report
    finally:
        simulation.close()

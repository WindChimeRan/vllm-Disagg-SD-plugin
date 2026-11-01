import os
import time
from pathlib import Path
from threading import Event

import pytest

from disagg_sd.transport import LocalWorker, ProcessWorker


def test_backpressure_and_failure_propagation():
    release = Event()

    def execute(batch):
        assert release.wait(5)
        raise ValueError("model failed")

    worker = LocalWorker(execute, 5)
    try:
        worker.submit((1, 2))
        assert worker.receive() is None
        with pytest.raises(RuntimeError, match="in flight"):
            worker.submit((3,))
        release.set()
        with pytest.raises(ValueError, match="model failed"):
            worker.receive(wait=True)
    finally:
        release.set()
        worker.close()


class _Executor:
    def __call__(self, batch):
        command, value = batch
        if command == "fail":
            raise ValueError("model failed")
        if command == "exit":
            os._exit(17)
        if command == "wait":
            while not Path(value).exists():
                time.sleep(0.01)
        if command == "write":
            Path(value).write_text("finished")
        return os.getpid(), value


def _fail_initialization():
    raise ValueError("initialization failed")


def test_process_isolation_backpressure_and_reuse(tmp_path):
    worker = ProcessWorker(_Executor, 5)
    release = tmp_path / "release"
    try:
        worker.submit(("wait", str(release)))
        assert worker.receive() is None
        with pytest.raises(RuntimeError, match="in flight"):
            worker.submit(("echo", 1))
        release.touch()
        pid, _ = worker.receive(wait=True)
        assert pid != os.getpid() and not worker.busy
        worker.submit(("echo", 42))
        assert worker.receive(wait=True) == (pid, 42)
        # Closing drains work, including side effects, before terminating.
        output = tmp_path / "output"
        worker.submit(("write", str(output)))
        worker.close()
        assert output.read_text() == "finished"
        with pytest.raises(RuntimeError, match="closed"):
            worker.submit(("echo", 1))
    finally:
        release.touch()
        worker.close()


@pytest.mark.parametrize("command", ["fail", "exit"])
def test_process_failure_does_not_silently_fall_back(command):
    worker = ProcessWorker(_Executor, 5)
    try:
        worker.submit((command, None))
        with pytest.raises(RuntimeError, match="model failed|without a result"):
            worker.receive(wait=True)
        assert not worker.busy and not worker._process.is_alive()
    finally:
        worker.close()


def test_process_initialization_failure():
    with pytest.raises(RuntimeError, match="initialization failed"):
        ProcessWorker(_fail_initialization, 5)


def test_process_timeout_releases_inputs_only_after_consumer_stops(tmp_path):
    worker = ProcessWorker(_Executor, 5)
    worker._timeout = 0.05  # Keep startup separate from the execute deadline.
    try:
        worker.submit(("wait", str(tmp_path / "never-created")))
        with pytest.raises(TimeoutError, match="timeout"):
            worker.receive(wait=True)
        assert not worker._process.is_alive() and not worker.busy
    finally:
        worker.close()

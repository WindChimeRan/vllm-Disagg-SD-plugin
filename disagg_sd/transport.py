"""Bounded execution with one in-flight batch and explicit completion.

The coordinator owns request state. The worker receives immutable inputs and
returns results through a completion channel; it never calls back into it.
"""

import multiprocessing as mp
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol


class Worker[Input, Output](Protocol):
    """Bounded submission and explicit completion; no coordinator callbacks."""

    @property
    def busy(self) -> bool: ...

    def submit(self, batch: Input) -> None: ...

    def receive(self, *, wait: bool = False) -> Output | None: ...

    def close(self) -> None: ...


class LocalWorker[Input, Output]:
    def __init__(self, execute: Callable[[Input], Output], timeout: float):
        self._execute = execute
        self._timeout = timeout
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="disagg-draft"
        )
        self._future: Future[Output] | None = None

    @property
    def busy(self) -> bool:
        return self._future is not None

    def submit(self, batch: Input) -> None:
        if self.busy:
            raise RuntimeError("Only one draft batch may be in flight")
        self._future = self._pool.submit(self._execute, batch)

    def receive(self, *, wait: bool = False) -> Output | None:
        future = self._future
        if future is None or (not wait and not future.done()):
            return None
        result = future.result(timeout=self._timeout)
        self._future = None
        return result

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._future = None


def _serve(connection) -> None:
    try:
        # Receive the initializer after spawn so multiprocessing's retained
        # process arguments do not hold model tensors for the worker's lifetime.
        initialize = connection.recv()
        execute = initialize()
        del initialize
        connection.send((True, None))
        while (batch := connection.recv()) is not None:
            result = execute(batch)
            # Release imported storage before acknowledging its producer.
            del batch
            connection.send((True, result))
            del result
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


class ProcessWorker[Input, Output]:
    """A spawned executor with independent Python and CUDA launch scheduling.

    The initializer and inputs must be picklable. Inputs remain owned by the
    producer until completion; the executor must finish using them before it
    returns. Results must be ordinary values, not CUDA tensors owned by the
    child. Model-specific serialization belongs outside this transport.
    """

    def __init__(
        self, initialize: Callable[[], Callable[[Input], Output]], timeout: float
    ):
        self._timeout = timeout
        self._input: Input | None = None
        self._closed = False
        context = mp.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_serve, args=(child,), daemon=True)
        self._process.start()
        child.close()
        try:
            self._connection.send(initialize)
            self._read()
        except BaseException:
            self._stop()
            raise

    @property
    def busy(self) -> bool:
        return self._input is not None

    def submit(self, batch: Input) -> None:
        if self._closed:
            raise RuntimeError("Draft worker is closed")
        if self.busy:
            raise RuntimeError("Only one draft batch may be in flight")
        if batch is None:
            raise ValueError("None is reserved for worker shutdown")
        self._input = batch
        try:
            self._connection.send(batch)
        except BaseException:
            self._stop()
            raise

    def _read(self) -> Output:
        if not self._connection.poll(self._timeout):
            raise TimeoutError("Draft worker did not respond before its timeout")
        try:
            ok, result = self._connection.recv()
        except EOFError as error:
            raise RuntimeError("Draft worker exited without a result") from error
        if not ok:
            raise RuntimeError(f"Draft worker failed:\n{result}")
        return result

    def receive(self, *, wait: bool = False) -> Output | None:
        if not self.busy:
            return None
        if not wait and not self._connection.poll():
            if not self._process.is_alive():
                self._stop()
                raise RuntimeError("Draft worker exited without a result")
            return None
        try:
            result = self._read()
        except BaseException:
            # On failure, stop the consumer before releasing shared inputs.
            self._stop()
            raise
        self._input = None
        return result

    def _stop(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(self._timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join()
        self._connection.close()
        self._input = None
        self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.receive(wait=True)
            self._connection.send(None)
            self._process.join(self._timeout)
        finally:
            self._stop()

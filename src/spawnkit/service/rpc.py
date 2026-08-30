"""What a batched RPC is: how N requests become one forward, and one result becomes N responses.

The service does not know what your model computes. It knows four things about each remote call,
and this module is where you declare them:

* **collate** — turn a list of per-request payloads into one batched input.
* **call** — run the model on that input, on the service's device.
* **split** — cut the batched output back into one response per request.
* **rows_in** — how many output rows a given request owns, so ``split`` knows where to cut.

Most calls are "concatenate these arrays along axis 0, run this method, slice the named output
fields back apart", and :class:`TensorRpc` is exactly that. Subclass :class:`Rpc` for the calls that
are not: a graph network whose batching is a library call rather than a concatenate, or a recurrent
net whose hidden state batches along axis 1 while everything beside it batches along axis 0. Both
are real, and a registry that only understood axes could express neither.

**RPCs are pickled to the service process**, which under ``spawn`` is the only way they get there.
That is why these are classes rather than a bundle of callables: a closure or a lambda cannot be
pickled, and a registry built from them fails at ``Process.start()`` with an error that names
pickling rather than the RPC. Subclass, keep your ``__init__`` arguments picklable, and this stays a
non-issue.

Responses travel as ``dict[str, np.ndarray]``, keyed by output field name. That is deliberate. The
obvious alternative — a positional tuple whose *length* tells the client which variant produced it —
works right up until two variants have the same arity, and then it fails by silently decoding one as
the other. A dict costs a few bytes per response and cannot do that.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from spawnkit.service.shared import SharedRowSpec

Payload = Any
"""Whatever one client sends for one call. Opaque to the service; only the RPC interprets it."""

Response = dict[str, np.ndarray]
"""What one client gets back: the output fields of its own rows, as numpy arrays."""


class Rpc:
    """One batched remote call. Subclass and implement the three steps that are not trivial.

    :param name: the wire name clients pass to
        :meth:`~spawnkit.service.client.ServiceClient.call`. Must be unique within a registry.
    :param shared: when set, this call uses the shared-memory transport instead of the queue, and
        :meth:`collate` / :meth:`split` are bypassed in favour of the row gather/scatter in
        :mod:`spawnkit.service.shared`. Note that transport's hard limit of one row per request.
    """

    def __init__(self, name: str, shared: SharedRowSpec | None = None) -> None:
        self.name = name
        self.shared = shared

    def collate(self, payloads: Sequence[Payload]) -> Any:
        """Turn this batch's per-request payloads into one batched input.

        Runs in the service process, on the CPU, before the device move. Receives the payloads in
        arrival order, and :meth:`split` must cut the result back apart in that same order.

        :param payloads: one entry per request in the batch.
        :return: whatever :meth:`call` accepts.
        """
        raise NotImplementedError

    def call(self, model: Any, batched: Any, device: torch.device) -> Any:
        """Run the model on the batched input, on ``device``.

        Responsible for its own device move: batching a graph and batching a tensor need different
        moves, and pretending otherwise would push a special case into the service.

        :param model: the service's device-local model.
        :param batched: the output of :meth:`collate`, or — on the shared-memory transport — a tuple
            of already-batched tensors in the spec's declared field order.
        :param device: where the model lives.
        :return: an object whose output fields are tensors.
        """
        raise NotImplementedError

    def split(self, outputs: Any, row_counts: Sequence[int]) -> list[Response]:
        """Cut the batched output into one response per request.

        Must return host-side numpy: a response crosses a process boundary, and a CUDA tensor cannot.

        :param outputs: whatever :meth:`call` returned.
        :param row_counts: how many rows each request owns, in the order :meth:`collate` saw them.
        :return: one response dict per request.
        """
        raise NotImplementedError

    def to_tensors(self, batched: Any, device: torch.device) -> tuple[torch.Tensor, ...] | None:
        """Return the batched input as device tensors, or ``None`` if this RPC has no such form.

        This is what makes an RPC eligible for CUDA-graph replay: a graph is captured against fixed
        input buffers, so the service must be able to see the inputs *as tensors* to copy them in.
        An RPC whose batched input is not a tuple of tensors — a batched graph object, say — returns
        ``None`` and is served eagerly, which is correct rather than a limitation.

        :param batched: the output of :meth:`collate`.
        :param device: where the model lives.
        :return: the tensors in call order, or ``None``.
        """
        del batched, device
        return None

    @property
    def graph_output_fields(self) -> tuple[str, ...]:
        """Return the output field names a captured graph must reproduce; empty disables replay."""
        return ()

    def rows_in(self, payload: Payload) -> int:
        """How many output rows this request owns. Override when a client may batch several.

        The default of 1 is right for a client that asks about one state at a time. A client that
        batches several into one call must get all of them back, and a wrong count here
        misattributes results *between clients* — silently, and in a way that looks like a model bug.

        :param payload: the request's payload.
        :return: its row count.
        """
        del payload
        return 1

    def __repr__(self) -> str:
        """Name the RPC and its transport."""
        transport = "shared" if self.shared is not None else "queue"
        return f"{type(self).__name__}(name={self.name!r}, transport={transport!r})"


class TensorRpc(Rpc):
    """The common case: concatenate arrays in, run a model method, slice named fields out.

    The payload is a tuple of numpy arrays, one per entry in ``input_axes``, concatenated along the
    axis given for each. Per-axis rather than one global axis because recurrent state genuinely
    batches differently from everything beside it — an LSTM's ``(h, c)`` is
    ``[layers, batch, hidden]`` and batches on axis 1 while the same call's inputs batch on axis 0.

    The model's output is read by attribute name (a namedtuple, dataclass, or anything with those
    attributes), each field sliced along ``row_axis``. A field whose value is ``None`` is dropped
    from the response rather than sent as ``None``, so an optional head costs nothing when absent.

    :param name: the RPC's wire name.
    :param method: the model method to call, by name. Looked up per call, so a weight resync that
        rebinds it is picked up. A callable is accepted too and is invoked as ``fn(model, *tensors)``
        — it must be picklable, so a module-level function rather than a lambda.
    :param input_axes: concatenation axis for each payload array, in payload order.
    :param output_fields: attribute names to read off the model's output.
    :param row_axis: axis along which output fields are sliced per request. A field whose rank is too
        low for ``row_axis`` is sliced on axis 0 instead — that is how an LSTM state (axis 1) and a
        value vector (axis 0) can come out of the same call.
    :param shared: see :class:`Rpc`.

    Examples
    --------
    >>> from spawnkit.service import TensorRpc
    >>> step = TensorRpc(
    ...     "step",
    ...     method="recurrent_inference",
    ...     input_axes=(0, 0),                    # hidden [n, H] and action [n, 1]
    ...     output_fields=("hidden_state", "reward", "policy", "value"),
    ... )
    >>> step
    TensorRpc(name='step', transport='queue')
    """

    def __init__(
        self,
        name: str,
        method: str | Callable[..., Any],
        input_axes: Sequence[int],
        output_fields: Sequence[str],
        row_axis: int = 0,
        shared: SharedRowSpec | None = None,
    ) -> None:
        super().__init__(name, shared=shared)
        self.method = method
        self.input_axes = tuple(input_axes)
        self.output_fields = tuple(output_fields)
        self.row_axis = row_axis

    def collate(self, payloads: Sequence[Payload]) -> tuple[np.ndarray, ...]:
        """Concatenate each payload position along its declared axis."""
        # zip(*payloads) regroups per-request tuples into per-argument tuples: the k-th entry holds
        # every request's k-th array, which is what concatenate needs.
        columns = list(zip(*payloads, strict=True))
        if len(columns) != len(self.input_axes):
            msg = (
                f"rpc {self.name!r}: payload has {len(columns)} arrays "
                f"but {len(self.input_axes)} axes were declared"
            )
            raise ValueError(msg)
        for index, payload in enumerate(payloads):
            self._check_rows_agree(payload, index)
        return tuple(
            np.concatenate(column, axis=axis)
            for column, axis in zip(columns, self.input_axes, strict=True)
        )

    def _check_rows_agree(self, payload: Payload, index: int) -> None:
        """Every array in one request must carry the same number of rows.

        Silent otherwise, and wrong in the worst way. ``rows_in`` reads the row count off the
        *first* array, so a request whose arrays disagree is split at the wrong boundaries and its
        rows are handed to the neighbouring clients — the misattribution this module warns about,
        arriving with no exception and no shape error, because concatenation along a batch axis
        happily accepts mismatched lengths.
        """
        counts = {
            int(np.asarray(array).shape[axis])
            for array, axis in zip(payload, self.input_axes, strict=True)
        }
        if len(counts) > 1:
            msg = (
                f"rpc {self.name!r}: request {index} has arrays of differing row counts "
                f"{sorted(counts)} along the declared axes {self.input_axes}. Every array in one "
                "request must describe the same rows, or the response is split at the wrong "
                "boundary and rows are returned to the wrong client."
            )
            raise ValueError(msg)

    def call(self, model: Any, batched: Sequence[Any], device: torch.device) -> Any:
        """Move the batched inputs to ``device`` and invoke the model method."""
        # as_tensor rather than from_numpy: the shared-memory transport hands this already-batched
        # tensors and the queue transport hands it numpy. One call site serves both.
        tensors = [torch.as_tensor(array).to(device) for array in batched]
        if isinstance(self.method, str):
            return getattr(model, self.method)(*tensors)
        return self.method(model, *tensors)

    def split(self, outputs: Any, row_counts: Sequence[int]) -> list[Response]:
        """Slice each declared output field into one response per request."""
        arrays = {}
        for field in self.output_fields:
            value = getattr(outputs, field, None)
            if value is not None:
                arrays[field] = value.detach().cpu().numpy()

        total_rows = sum(row_counts)
        responses: list[Response] = []
        start = 0
        for count in row_counts:
            stop = start + count
            responses.append({
                field: _take_rows(array, start, stop, self.row_axis, total_rows)
                for field, array in arrays.items()
            })
            start = stop
        return responses

    def to_tensors(self, batched: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Return the concatenated arrays as device tensors, in call order."""
        return tuple(torch.as_tensor(array).to(device) for array in batched)

    @property
    def graph_output_fields(self) -> tuple[str, ...]:
        """Return the declared output fields; a captured graph reproduces exactly these."""
        return self.output_fields

    def rows_in(self, payload: Payload) -> int:
        """Return the first array's extent along its declared axis; every array in a payload agrees."""
        return int(np.asarray(payload[0]).shape[self.input_axes[0]])


def _take_rows(array: np.ndarray, start: int, stop: int, axis: int, total_rows: int) -> np.ndarray:
    """Slice ``array[start:stop]`` along whichever axis actually holds the batch.

    One call can return fields that batch on different axes — an ``[layers, batch, hidden]``
    recurrent state batches on axis 1 while a ``[batch, values]`` head beside it batches on axis 0 —
    so ``row_axis`` is a preference, not an instruction, and the batch axis is *identified* rather
    than assumed.

    Identified by extent: the declared ``row_axis`` is used only when the field is actually that long
    along it. Rank alone is not enough and choosing by rank was a bug — a ``[batch, values]`` head has
    rank 2, so a ``row_axis`` of 1 sliced it along ``values``, handing every request the whole batch
    and silently dropping columns from the last one. No exception, no shape error at the client: the
    exact "misattributes results between clients" failure this module warns about.

    When both candidate axes have length ``total_rows`` the declaration wins, since nothing else can
    break the tie. If your outputs are genuinely ambiguous that way, subclass :class:`Rpc` and write
    ``split`` yourself rather than relying on a heuristic to guess right.
    """
    prefers_declared = array.ndim > axis and array.shape[axis] == total_rows
    effective = axis if prefers_declared else 0
    index: list[Any] = [slice(None)] * array.ndim
    index[effective] = slice(start, stop)
    return array[tuple(index)]

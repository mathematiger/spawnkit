"""The shared-memory transport: one pre-allocated row per client, instead of pickling every call.

The queue transport pickles a request's tensors, pushes the bytes through a pipe, and unpickles them
on the other side — twice per round trip. For small, frequent calls that serialisation is a
meaningful share of the latency, and it is pure overhead: the data is small and its shape never
changes.

This transport removes it. Every client owns **one row** in each request and response buffer, indexed
by its client id, and the queue then carries only ``(client_id, request_id, rpc_name, None)`` — an
identifier, no payload. The service index-gathers the batch's rows into one tensor, runs the forward,
index-scatters the outputs back, and signals each client.

Why one row per client is safe without a lock: the service enforces **one in-flight request per
client**, so a client's row is owned exclusively by that client between submitting a request and
receiving its response, and by the service in between. No two writers ever hold the same row.

That same invariant is the transport's limit, and it is not a soft one:

* **A request carrying more than one row cannot use it.** A client that batches several states into
  one call needs several rows, and it has one. Such an RPC must stay on the queue transport.
* **Shapes are fixed at allocation.** Every row of a field has the same shape and dtype, declared up
  front. A call whose tensor width varies per request belongs on the queue.
* **Clients must clone what they read.** The response row is reused by the client's next call, so a
  caller that keeps a reference — into a cache, a search tree, a replay buffer — keeps a reference to
  a buffer that is about to be overwritten. :class:`~spawnkit.service.client.ServiceClient` clones on
  read for exactly this reason.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

FieldSpec = Mapping[str, tuple[tuple[int, ...], torch.dtype]]
"""``{field_name: (per_row_shape, dtype)}`` — the layout of one client's row in one buffer."""


@dataclass(frozen=True)
class SharedRowSpec:
    """Declares which fields of an RPC travel through shared memory rather than the queue.

    Attach one to an :class:`~spawnkit.service.rpc.Rpc` to move that call onto the shared-memory
    transport. Both sides read the declaration, so the client and the service cannot disagree about
    the layout — which is the failure mode a hand-rolled shared buffer has, and it corrupts data
    rather than raising.

    :param request: fields the client writes before submitting, in the positional order the RPC's
        ``call`` expects them. Python dicts preserve insertion order, and that order is the contract.
    :param response: fields the service writes back, by name. Read off the model's output object by
        attribute, so the names must match what the forward returns.

    Examples
    --------
    >>> import torch
    >>> from spawnkit.service import SharedRowSpec
    >>> spec = SharedRowSpec(
    ...     request={"hidden": ((128,), torch.float32), "action": ((1,), torch.long)},
    ...     response={"value": ((1,), torch.float32), "policy": ((60,), torch.float32)},
    ... )
    >>> list(spec.request)
    ['hidden', 'action']
    """

    request: FieldSpec
    response: FieldSpec


class SharedRows:
    """The allocated shared-memory buffers: ``num_clients`` rows per declared field.

    Allocate once in the parent, **before** any worker spawns, and pass the same object to the
    service and to every client. torch's shared memory survives the spawn pickle as a handle to the
    same storage, which is the whole point; constructing one per process would give each of them a
    private copy that silently never agrees with the others.

    :param num_clients: how many rows each buffer holds. Client ids index into it, so this is the
        highest client id plus one.
    :param spec: the field layout.

    Examples
    --------
    >>> import torch
    >>> from spawnkit.service import SharedRows, SharedRowSpec
    >>> spec = SharedRowSpec(
    ...     request={"x": ((4,), torch.float32)}, response={"y": ((2,), torch.float32)},
    ... )
    >>> rows = SharedRows(num_clients=3, spec=spec)
    >>> rows.request["x"].shape
    torch.Size([3, 4])
    >>> rows.response["y"].is_shared()
    True
    """

    def __init__(self, num_clients: int, spec: SharedRowSpec) -> None:
        self.num_clients = int(num_clients)
        self.spec = spec
        self.request = {
            name: torch.zeros(self.num_clients, *shape, dtype=dtype).share_memory_()
            for name, (shape, dtype) in spec.request.items()
        }
        self.response = {
            name: torch.zeros(self.num_clients, *shape, dtype=dtype).share_memory_()
            for name, (shape, dtype) in spec.response.items()
        }

    def write_request(self, client_id: int, payload: Mapping[str, torch.Tensor]) -> None:
        """Copy one client's request values into its row. Called by the client, before submitting.

        :param client_id: the row to write.
        :param payload: ``{field: tensor}`` for every field in the spec's ``request``. A tensor is
            reshaped to the row's shape, so a ``[1, H]`` batch-of-one and a ``[H]`` vector are both
            accepted — clients disagree about that and neither is wrong.
        :raises KeyError: if a declared request field is missing from ``payload``.
        :raises ValueError: if a value does not hold exactly one row's worth of elements — most often
            a multi-row request, which this transport structurally cannot carry.
        """
        for name, row in self.request.items():
            value = payload[name].detach()
            row_shape = row.shape[1:]
            if value.numel() != row[client_id].numel():
                msg = (
                    f"shared-memory field {name!r} takes one row shaped {tuple(row_shape)}, but "
                    f"{value.numel()} values were given. This transport allocates exactly one row "
                    "per client, so a request carrying several rows cannot use it: drop this RPC's "
                    "SharedRowSpec to move it onto the queue transport, which has no such limit."
                )
                raise ValueError(msg)
            row[client_id].copy_(value.reshape(row_shape))

    def read_response(self, client_id: int) -> dict[str, torch.Tensor]:
        """Return a **cloned** copy of one client's response row, as ``{field: tensor}``.

        The clone is not optional. The row is reused by this client's next call, so handing out a
        view would hand out a buffer that changes under the caller — the resulting corruption is
        silent, delayed, and looks like a model bug rather than a transport one.

        :param client_id: the row to read.
        :return: one tensor per response field, shaped ``[1, *field_shape]`` so it matches what the
            queue transport returns for a single-row request.
        """
        return {name: row[client_id].clone().unsqueeze(0) for name, row in self.response.items()}

    def gather_request(self, client_ids: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Collect the batch's request rows into one tensor per field, on ``device``.

        :param client_ids: the ids of the requests in this batch, in order.
        :param device: where the service's model lives.
        :return: one batched tensor per request field, in the spec's declared order.
        """
        return tuple(row.index_select(0, client_ids).to(device) for row in self.request.values())

    def scatter_response(self, client_ids: torch.Tensor, outputs: object) -> None:
        """Write the model's outputs back into each requesting client's response row.

        Fields are read off ``outputs`` by attribute and reshaped to the row layout, so a head that
        returns ``[n]`` and one that returns ``[n, 1]`` both land correctly. A field the model did
        not produce is left at whatever the row held, which for an optional head means the zeros it
        was allocated with.

        :param client_ids: the ids of the requests in this batch, in the same order as the outputs.
        :param outputs: the model's output object.
        """
        count = int(client_ids.shape[0])
        for name, row in self.response.items():
            value = getattr(outputs, name, None)
            if value is None:
                continue
            row.index_copy_(
                0,
                client_ids,
                value.detach().cpu().reshape(count, *row.shape[1:]).to(row.dtype).contiguous(),
            )

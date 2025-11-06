import asyncio
import http
import logging
import os
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def _log_policy_timing(self, step: int, timing: dict) -> None:
        """Log policy timing information in a formatted, hierarchical style."""
        def fmt(key: str) -> str:
            if key not in timing:
                return None
            v = timing[key]
            if isinstance(v, float):
                return f"{v:7.2f}"
            return str(v)

        def fmt_float(key: str) -> float:
            if key not in timing:
                return 0.0
            v = timing[key]
            if isinstance(v, float):
                return v
            return 0.0

        # Core timing metrics
        lines = [
            f"[PROFILE] Step {step:4d} | Policy Timing",
            f"{'─' * 70}"
        ]

        # Processing steps with indentation
        steps_info = []
        if copy_ms := fmt("copy_input_ms"):
            steps_info.append(f"  copy_input:          {copy_ms} ms")
        if input_ms := fmt("input_transform_ms"):
            steps_info.append(f"  input_transform:     {input_ms} ms")
        if batch_ms := fmt("batch_to_device_ms"):
            steps_info.append(f"  batch_to_device:     {batch_ms} ms")
        if obs_ms := fmt("observation_build_ms"):
            steps_info.append(f"  observation_build:   {obs_ms} ms")

        # Token profiling overhead (if available)
        prefix_orig_ms = fmt("prefix_orig_ms")
        prefix_ms = fmt("prefix_ms")
        if prefix_orig_ms or prefix_ms:
            steps_info.append(f"  ├─ token_profiling: (profiling overhead, not part of inference)")
            if prefix_orig_ms:
                steps_info.append(f"  │  ├─ calc_orig_len:  {prefix_orig_ms} ms")
            if prefix_ms:
                steps_info.append(f"  │  └─ embed_prefix:   {prefix_ms} ms")

        if infer_ms := fmt("infer_ms"):
            steps_info.append(f"  [MODEL_INFER]:       {infer_ms} ms  ← Core computation")
        # Optional pruning diagnostics
        if prune_over := fmt("prune_overhead_ms"):
            steps_info.append(f"  prune_overhead:      {prune_over} ms")
        if infer_wo_prune := fmt("infer_wo_prune_ms"):
            steps_info.append(f"  infer_wo_prune:      {infer_wo_prune} ms")
        if host_ms := fmt("host_copy_ms"):
            steps_info.append(f"  host_copy:           {host_ms} ms")
        if output_ms := fmt("output_transform_ms"):
            steps_info.append(f"  output_transform:    {output_ms} ms")

        if steps_info:
            lines.extend(steps_info)

        # Show total and accounting
        if total_ms := fmt("total_ms"):
            lines.append(f"  {'-' * 40}")
            lines.append(f"  [TOTAL]:             {total_ms} ms")

            # Verify accounting
            accounted = (
                fmt_float("copy_input_ms") +
                fmt_float("input_transform_ms") +
                fmt_float("batch_to_device_ms") +
                fmt_float("observation_build_ms") +
                fmt_float("prefix_orig_ms") +  # Profiling overhead
                fmt_float("prefix_ms") +        # Profiling overhead
                fmt_float("infer_ms") +
                fmt_float("host_copy_ms") +
                fmt_float("output_transform_ms")
            )
            total_float = fmt_float("total_ms")
            unaccounted = total_float - accounted
            if abs(unaccounted) > 0.5:
                lines.append(f"  unaccounted:         {unaccounted:7.2f} ms ⚠️")

        # Token statistics summary (always show if available)
        if "prefix_orig_len" in timing and "prefix_len" in timing:
            orig_len = int(timing.get("prefix_orig_len", 0))
            curr_len = int(timing.get("prefix_len", 0))
            keep_ratio = timing.get("prefix_keep_ratio", 0) * 100
            lines.append(f"  {'-' * 40}")
            lines.append(f"  Tokens: {orig_len:4d} → {curr_len:4d} ({keep_ratio:5.1f}% kept)")

        # Pruning info (only when enabled)
        if timing.get("prune_enabled"):
            ratio = timing.get("prune_ratio", 0)
            lines.append(f"  [PRUNING]: ENABLED | ratio={ratio:.2f}")

        lines.append(f"{'─' * 70}")

        # Log all lines
        for line in lines:
            logger.info(line)

    def _log_server_timing(self, step: int, timing: dict) -> None:
        """Log server timing information in a formatted style."""
        def fmt(key: str) -> str:
            if key not in timing:
                return "    -"
            return f"{timing[key]:7.2f}"

        def fmt_float(key: str) -> float:
            if key not in timing:
                return 0.0
            return float(timing.get(key, 0))

        recv = fmt("recv_ms")
        unpack = fmt("unpack_ms")
        infer = fmt("infer_ms")
        pack = fmt("pack_ms")
        send = fmt("send_ms")
        total = fmt("total_ms")

        lines = [
            f"[PROFILE] Step {step:4d} | Server Timing",
            f"{'─' * 70}",
            f"  recv:              {recv} ms  (WebSocket receive)",
            f"  unpack:            {unpack} ms  (msgpack deserialize)",
            f"  infer:             {infer} ms  (Policy.infer call)",
            f"  pack:              {pack} ms  (msgpack serialize)",
            f"  send:              {send} ms  (WebSocket send)",
            f"  {'-' * 40}",
            f"  [TOTAL]:           {total} ms"
        ]

        # Verify accounting (total should ≈ recv+unpack+infer+pack, send is measured separately)
        accounted = (
            fmt_float("recv_ms") +
            fmt_float("unpack_ms") +
            fmt_float("infer_ms") +
            fmt_float("pack_ms")
        )
        total_float = fmt_float("total_ms")
        unaccounted = total_float - accounted
        if abs(unaccounted) > 0.5:
            lines.append(f"  unaccounted:       {unaccounted:7.2f} ms ⚠️")

        # Optional: show previous iteration timing if available
        if "prev_total_ms" in timing:
            prev_total = fmt("prev_total_ms")
            prev_send = fmt("prev_send_ms")
            lines.append(f"  {'-' * 40}")
            lines.append(f"  prev_iteration: {prev_total} ms (incl. send: {prev_send} ms)")

        lines.append(f"{'─' * 70}")

        for line in lines:
            logger.info(line)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        prev_send_time = None
        step_counter = 0  # Track request number for better profiling output
        while True:
            try:
                # Receive: do not include idle wait time in server totals.
                # We start timing after the payload has been received to avoid overlap with client timing window.
                raw = await websocket.recv()
                # Define recv_time as 0.0 to indicate no wait counted; alternatively could capture negligible buffer copy.
                recv_time = 0.0

                start_time = time.monotonic()

                # Unpack
                t0 = time.monotonic()
                obs = msgpack_numpy.unpackb(raw)
                unpack_time = time.monotonic() - t0

                # Policy inference
                t0 = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - t0

                # Log policy timing if profiling is enabled (formatted + hierarchical)
                if policy_timing := action.get("policy_timing"):
                    self._log_policy_timing(step_counter, policy_timing)

                # Serialize once to estimate pack cost (without timing included)
                t0 = time.monotonic()
                payload = packer.pack(action)
                pack_time = time.monotonic() - t0

                # Define server total as the sum of measured components (recv+unpack+infer+pack),
                # excluding send time so totals align with breakdown and client comparisons.
                total_time = recv_time + unpack_time + infer_time + pack_time

                action["server_timing"] = {
                    "recv_ms": recv_time * 1000.0,
                    "unpack_ms": unpack_time * 1000.0,
                    "infer_ms": infer_time * 1000.0,
                    "pack_ms": pack_time * 1000.0,
                    "total_ms": total_time * 1000.0,
                }
                if prev_total_time is not None:
                    # Report previous iteration totals including send time, and previous send specifically.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0
                    if prev_send_time is not None:
                        action["server_timing"]["prev_send_ms"] = prev_send_time * 1000.0

                # Pack again including timing, then send and measure send time
                t0 = time.monotonic()
                payload = packer.pack(action)
                pack_time_with_timing = time.monotonic() - t0  # not recorded

                t0 = time.monotonic()
                await websocket.send(payload)
                send_time = time.monotonic() - t0

                # Update send time once measured
                action["server_timing"]["send_ms"] = send_time * 1000.0

                # Optional logging of server timing when profiling is enabled
                if os.getenv("OPENPI_INFER_PROFILE") not in (None, "", "0", "false", "False", "no"):
                    self._log_server_timing(step_counter, action["server_timing"])

                # For the next iteration, record totals including send time.
                prev_total_time = total_time + send_time
                prev_send_time = send_time
                step_counter += 1  # Increment step counter after successful inference

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None

# RoCE all-gather for two ranks

One-shot all-gather between two DGX Sparks over RoCE: the GPU side of b12x's "RoCEnante" exchange (https://github.com/local-inference-lab/b12x, b12x/comm/roce, Apache License 2.0), written for two ranks. The host side (zig/src/cuda/roce.zig) owns a pinned region both GPUs and both NICs reach:

```
  recv[slot]       the peer's shard, written by the peer's NIC (2 slots of slot_bytes)
  flag[slot][hca]  the sequence the peer's NIC writes after its stripe on that HCA (128 bytes apart)
  send[slot]       this rank's shard, staged here for its NIC (2 slots)
  ctrl             {seq, nbytes of slot 0, nbytes of slot 1, error seq, missing hca}: the proxy's doorbell
```

One launch: (1) every block stages its share of the shard into send[seq & 1]; (2) the last block to stage publishes the slot's size and rings ctrl.seq after a system fence, and the proxy thread RDMA-writes the shard and then a flag per HCA stripe on the same queue pair (so a flag never lands before its stripe); (3) each block waits for every stripe's flag; (4) every block writes both shards in rank order (the layout NCCL's all-gather gives); (5) the last block to finish advances the device epoch, so a CUDA graph replays the exchange with new sequences. A wait past the configured wall-clock deadline poisons the exchange: later launches return at once and the host reports the error.

## Host side

Small all-gathers between two ranks as one-shot RDMA writes over RoCE: b12x's "RoCEnante" exchange (https://github.com/local-inference-lab/b12x, b12x/comm/roce, Apache License 2.0), written for two ranks: the kernel (kernels/cuda/roce_gather.cu) stages this rank's shard in pinned host memory and rings a doorbell; a proxy thread here RDMA-writes it to the peer, striped over the HCAs, each stripe followed by its sequence flag on the same queue pair; the kernel waits for the peer's flags and copies both shards out. A decode step's gathers are a few rows of partials. On two GB10s with one ConnectX-7 cable (NCCL 2.30.7, `tf-cuda-test roce-gather`, 2,000 calls back to back) this path took 10.4 us a call at 4 KiB a rank against NCCL's 12.3, 11.6 against 13.5 at 8 KiB, 17.4 against 20.5 at 32 KiB, and 17.4 against 51.5 at 64 KiB. libibverbs is opened at run time (as the driver and NCCL are): without it, or without an active HCA, the caller keeps NCCL.

## Setup and use

`tp_group.Group.openRoce` brings the path up on a two-rank group: each rank opens its HCAs, and what each brings (its region, keys, queue pairs and the largest shard it would send) crosses the group's own NCCL communicator, as does each rank's verdict after connecting. Both ranks get the path or neither does (`TF_ROCE=0` on either rank keeps NCCL). The `Gather` is bound to the stream passed to `openRoce`; gathers on other streams use NCCL. RoCE's sequence number comes from the device epoch written by the prior gather, so overlapping gathers on separate streams could read one epoch and reuse a slot. `Group.allGather` sends aligned gathers of up to the smaller rank's cap (glm_dsa opens 4 MiB) by RoCE when issued on the bound stream and leaves larger, odd-sized, or other-stream gathers to NCCL. `Group.gather` (setup handshakes such as the switch and expert-rows agreements and side-communicator ids) always uses NCCL, which waits without a deadline: in the native server the ranks load weights at different speeds after RoCE is up, and a RoCE handshake would hit the peer deadline (2 s under `TF_GLM53_COLLECTIVE=roce`). A rank whose buffers are not 16-byte aligned gets `error.RoceAlignmentViolation` instead of falling back alone. Liveness is `tp_watch`'s, as for every group.

A failed exchange fills gathered rows with `0xffffffff` words. This is NaN in fp32 and bf16/fp16, and -1 in signed 32-bit positions. Call `Group.syncChecked` before consuming a gathered result; it synchronizes, checks the sticky RoCE status, and aborts the group on failure.

The all-gather path remains enabled by default unless `TF_ROCE=0`. `TF_ROCE_TIMEOUT_MS` can set a shared 1..3,600,000 ms deadline; plain gathers default to 120,000 ms to preserve the pre-opt-in wait behavior. `TF_GLM53_COLLECTIVE=roce` enables the RoCE all-reduce selector and defaults to 2,000 ms. Both ranks must agree on the timeout and settings. Poison is terminal for a `Gather`; a complete deinit and re-init clears it because initialization zeroes the mapped control region and device counters. Restart the TP pair after a failed exchange.

Adaptive expert rows (`TF_GLM53_EXPERT_ROWS_ADAPT`) work with RoCE: RoCE registers only its own pinned host region, and the kernel reaches the VMM-leased adapt workspace as ordinary device memory. Startup refuses with `error.ExpertRowsAdaptRoceMisaligned` if any adapt workspace pointer is not 16-byte aligned.

`tf-cuda-test roce-gather <rank>` runs on both ranks (NCCL_COMM_ID set): every size up to the cap and past it against NCCL's all-gather, byte for byte, then each path's time a call.

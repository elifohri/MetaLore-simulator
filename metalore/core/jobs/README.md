# Jobs Module

Models the full lifecycle of a computation task from generation at an edge device to completion at a MEC server.


## Job Lifecycle

Each job passes through two sequential phases:

1. **Transmission** — sent from the entity's `TxQueue` over the wireless channel to the BS.
2. **Processing** — computed at the MEC server from its `ProcessQueue`.

Both phases support **partial work**: a job too large to complete in one timestep continues from where it left off in the next.


## Components

### `Job` — [`job.py`](job.py)

Represents a single task with immutable identity fields and mutable lifecycle state.

**Identity (read-only)**
| Field | Description |
|---|---|
| `id` | Unique job identifier |
| `entity_id` | ID of the generating entity |
| `entity_type` | `'UE'` or `'SENSOR'` |
| `data_size` | Total bits to transmit |
| `compute_size` | Total CPU cycles to process |
| `generated_at` | Timestep when the job was created |

**Lifecycle state (mutable)**
| Field | Description |
|---|---|
| `bits_remaining` | Bits not yet transmitted |
| `cycles_remaining` | Cycles not yet processed |
| `is_transmitted` | Set to `True` when transmission completes |
| `is_processed` | Set to `True` when processing completes |
| `tx_start_at / tx_end_at` | Transmission phase timestamps |
| `proc_start_at / proc_end_at` | Processing phase timestamps |

**Derived latency properties**
- `tx_queue_wait` — timesteps waiting before transmission started
- `tx_duration` — timesteps spent transmitting
- `proc_queue_wait` — timesteps waiting before processing started
- `proc_duration` — timesteps spent processing
- `total_latency` — end-to-end latency from generation to completion

---

### `TxQueue` / `ProcessQueue` — [`queue.py`](queue.py)

FIFO queues modelling the two waiting stages in the job lifecycle.

| Queue | Location | Holds |
|---|---|---|
| `TxQueue` | Per UE / Sensor | Jobs awaiting wireless transmission |
| `ProcessQueue` | Per BS | Jobs awaiting MEC computation |

Both share a common `JobQueue` base (`enqueue`, `head`, `dequeue`, `clear`, `length`).
`TxQueue` exposes `total_bits`; `ProcessQueue` exposes `total_cycles`.

---

### `JobGenerator` — [`generator.py`](generator.py)

Provides the two distributions used for job generation:

```python
generator.bernoulli(p)      # True/False — whether to generate a UE job
generator.poisson(lam)      # Poisson-sampled job size, minimum 1
```

The triggering logic (Bernoulli for UEs, periodic for sensors) and job construction live in `base.py`.


---

### `transmit()` — [`transfer.py`](transfer.py)

Drains bits from the head of a `TxQueue` at the allocated datarate.

```python
bits_sent, completed_jobs = transmit(queue, datarate, timestep=t)
```

- Partial transmission is supported: a large job spans multiple timesteps.
- Returns the list of fully transmitted jobs, ready to be enqueued at the BS.
- Sets `tx_start_at`, `tx_end_at` and `is_transmitted` on completed jobs.

---

### `process()` — [`processor.py`](processor.py)

Consumes CPU cycles from the head of a `ProcessQueue` at the allocated compute rate.

```python
cycles_consumed, completed_jobs = process(queue, compute_capacity, timestep=t)
```

- Partial processing is supported: a heavy job spans multiple timesteps.
- Returns the list of fully transmitted jobs, ready to be enqueued at the BS.
- Sets `proc_start_at`, `proc_end_at` and `is_processed` on completed jobs.

---

### `JobTracker` — [`tracker.py`](tracker.py)

Accumulates job statistics at four levels:

| Level | Example attributes |
|---|---|
| Episode totals | `total_generated`, `total_bits_transmitted` |
| Per-step totals | `step_generated`, `step_cycles_processed` |
| Per-entity episode | `entity_transmitted[entity_id]`, `entity_bits_transmitted[entity_id]` |
| Per-entity per-step | `step_entity_processed[entity_id]`, `step_entity_bits_transmitted[entity_id]` |

Call `begin_step()` at the start of each timestep to reset the step-level counters.
Call `reset()` at the start of each episode.

`to_dataframe()` returns a per-job DataFrame with all lifecycle columns, useful for post-episode latency analysis.
`save_log(path)` writes it to CSV.

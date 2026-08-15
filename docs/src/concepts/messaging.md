# Messaging

_Requires the `messaging` feature._

Agents that influence one another do so through **messages**. A message type is
registered against a **specialisation** that decides how messages are indexed and
delivered; producers emit messages during one stage and consumers read them in a
later stage, with the message buffers acting as a boundary between them.

## The four specialisations

| Specialisation | Delivery model | Use when |
| --- | --- | --- |
| **Brute-force** | Every consumer can see every message. | The set is small, or every agent must consider every message. |
| **Bucket** | Messages are grouped into a fixed number of buckets. | Messages are addressed to one of a known set of groups (a sector, a bank). |
| **Spatial** | Messages are indexed by grid cell. | Delivery is by locality — neighbours within a cell or radius. |
| **Targeted** | Messages are addressed to a specific recipient. | Each message has a single intended recipient. |

Each specialisation is registered with the matching `ModelBuilder` method
(`register_brute_force_message`, `register_bucket_message`,
`register_spatial_message`, `register_targeted_message`), which returns a typed
[`MessageHandle`]. The handle is a small `Copy` token a system stores and uses to
emit or read that message type.

## Emitting and consuming

A system emits with the message boundary's `emit`, or takes an `emitter` for a
tight loop that produces many messages without re-locking per message. Emission
during a parallel stage is staged per worker and finalised deterministically at
the stage boundary, so the delivered set does not depend on thread scheduling.

A consuming system reads the finalised messages for its handle in a later stage.
Because the message boundary owns a channel, the scheduler orders emitters before
consumers automatically.

## GPU messaging

With `messaging_gpu`, message buffers can live on the GPU so a GPU system both
produces and consumes messages without a round trip to the CPU. See [add a GPU
component and system](../how-to/gpu.md).

For worked examples of each specialisation, see [use each message
specialisation](../how-to/messaging.md).

[`MessageHandle`]: https://docs.rs/syren/latest/syren/messaging/struct.MessageHandle.html

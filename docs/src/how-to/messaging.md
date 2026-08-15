# Use each message specialisation

_Requires the `messaging` feature._

Register a message type against the specialisation that matches how it is
delivered, store the returned [`MessageHandle`], emit during one stage, and read
in a later one. The four registration methods take `&mut self` on `ModelBuilder`
and a [`Capacity`] hint (for example `Capacity::unbounded(64)`). Each message type
implements the trait for its specialisation.

## Brute-force

Every consumer can see every message. Register with `register_brute_force_message`:

```rust,ignore
let offers = builder.register_brute_force_message::<Offer>(Capacity::unbounded(64))?;
```

Use it when the message set is small or every agent must consider every message.

## Bucket

Messages are grouped into a fixed number of buckets, addressed by bucket index.
Register with `register_bucket_message`, giving the bucket count first:

```rust,ignore
let by_sector = builder.register_bucket_message::<Order>(num_sectors, Capacity::unbounded(256))?;
```

Use it when messages are addressed to one of a known set of groups — a sector, a
bank, a market.

## Spatial

Messages are indexed by grid cell, so a consumer reads the messages near a
position. Register with `register_spatial_message`, passing a spatial
configuration that describes the grid first:

```rust,ignore
let signals = builder.register_spatial_message::<Signal>(spatial_config, Capacity::unbounded(128))?;
```

Use it when delivery is by locality. The spatial index uses the same cell math as
the [space](../concepts/environments-space.md) layer.

## Targeted

Each message names a single recipient. Register with `register_targeted_message`:

```rust,ignore
let payments = builder.register_targeted_message::<Payment>(Capacity::unbounded(64))?;
```

Use it when a message has exactly one intended recipient.

## Emit and consume

Emit through the message boundary using the handle. For a loop that produces many
messages, take an `emitter` once rather than emitting one at a time:

```rust,ignore
// occasional emission
messages.emit(offers, Offer { /* ... */ })?;
```

A consuming system in a later stage reads the finalised messages for its handle.
Because the message boundary owns a channel, the scheduler orders emitters before
consumers automatically, and the delivered set is independent of thread
scheduling.

For GPU-resident messages, see [add a GPU component and
system](gpu.md).

[`MessageHandle`]: https://docs.rs/syren/latest/syren/messaging/struct.MessageHandle.html
[`Capacity`]: https://docs.rs/syren/latest/syren/messaging/struct.Capacity.html

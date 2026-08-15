# Order systems with channels

Access conflicts order systems that touch the same components. When two systems
touch *different* components but one must still run after the other — B consumes
an effect A produced — express that ordering with a **channel**.

## Channel IDs

A channel is identified by a `ChannelID`. Model resources own these; they are
not created directly. An environment key, a message handle, and a space handle
each own a channel, and a model can register dedicated **phase keys** used only
for ordering. Read the id from the resource, for example `key.channel_id()` or
`handle.channel_id()`.

## Declaring produce and consume

A system declares channels through its [`AccessSets`]: insert a channel id into
`produces` or `consumes`. Build the access set explicitly and construct the
system with `FnSystem::new`:

```rust,ignore
let mut access = AccessSets::default();
access.read.set(firm_id);
access.write.set(bank_id);
access.produces.insert(phases.credit_cleared.channel_id());

let system = FnSystem::new(id, "credit_market", access, move |ecs| {
    // ...
    Ok(())
});
```

A later system that must run after it consumes the same channel:

```rust,ignore
let mut access = AccessSets::default();
access.read.set(bank_id);
access.consumes.insert(phases.credit_cleared.channel_id());
```

The scheduler places every consumer of a channel in a stage after all of its
producers. Systems that neither produce nor consume the channel are unaffected
and may still parallelise with either side.

## Ordering whole phases

To sequence phases of a tick — "aggregate, then set targets, then run the labour
market" — give each phase a channel: the phase's systems produce it, and the next
phase's systems consume it. The macroeconomy example orders its quarterly
schedule this way; see `examples/macroeconomy/systems.rs`.

## Deriving component access and adding channels

`FnSystem::from_queries` derives component access from queries but does not add
channels. When a system needs both, build an `AccessSets` (setting the component
reads and writes, or deriving them), insert the channels, and use `FnSystem::new`.

[`AccessSets`]: https://docs.rs/syren/latest/syren/struct.AccessSets.html

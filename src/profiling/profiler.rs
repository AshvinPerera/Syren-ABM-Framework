//! Chrome Trace (“flame style”) profiling.
//!
//! Feature-gated with `--features profiling`.
//!
//! Usage:
//!   abm_framework::profiler::init("profile/trace.json");
//!   {
//!     let _g = abm_framework::profiler::span("Scheduler::run");
//!     // run ECS...
//!   }
//!   abm_framework::profiler::shutdown();

use std::borrow::Cow;
use std::path::{Path};
use std::fmt;

#[cfg(feature = "profiling")]
    mod enabled {
    use std::cell::RefCell;
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::path::{PathBuf};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    use super::*;


    /// A Chrome trace “complete event” (`ph:"X"`) plus optional metadata events (`ph:"M"`).
    #[derive(Debug)]
    enum TraceEvent {
        Complete {
            name: String,
            ts_us: u64,
            dur_us: u64,
            pid: u32,
            tid: u64,
            args: Vec<(String, ArgValue)>,
        },
        ThreadName {
            ts_us: u64,
            pid: u32,
            tid: u64,
            name: String,
        },
    }

    #[derive(Debug)]
    enum ArgValue {
        Str(String),
        U64(u64),
        I64(i64),
        F64(f64),
        Bool(bool),
    }

    // ✅ Conversion lives INSIDE enabled, so it can return private ArgValue
    impl super::Arg {
        fn into_enabled(self) -> ArgValue {
            match self {
                super::Arg::Str(s) => ArgValue::Str(s),
                super::Arg::U64(v) => ArgValue::U64(v),
                super::Arg::I64(v) => ArgValue::I64(v),
                super::Arg::F64(v) => ArgValue::F64(v),
                super::Arg::Bool(v) => ArgValue::Bool(v),
            }
        }
    }

    impl ArgValue {
        fn write_json<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
            match self {
                ArgValue::Str(s) => write_json_string(w, s),
                ArgValue::U64(v) => write!(w, "{}", v),
                ArgValue::I64(v) => write!(w, "{}", v),
                ArgValue::F64(v) => {
                    if v.is_finite() {
                        write!(w, "{}", v)
                    } else {
                        write_json_string(w, &format!("{v}"))
                    }
                }
                ArgValue::Bool(v) => write!(w, "{}", if *v { "true" } else { "false" }),
            }
        }
    }

    struct ProfilerState {
        start: Instant,
        out_path: PathBuf,
        pid: u32,
        is_on: AtomicBool,
        events: Mutex<Vec<TraceEvent>>,
    }

    static STATE: OnceLock<ProfilerState> = OnceLock::new();
    static NEXT_TID: AtomicU64 = AtomicU64::new(1);

    thread_local! {
        static TID: u64 = NEXT_TID.fetch_add(1, Ordering::Relaxed);
        static PENDING_ARGS: RefCell<Vec<(String, ArgValue)>> = const { RefCell::new(Vec::new()) };
    }

    fn now_us() -> u64 {
        let st = STATE.get().expect("profiler::init() must be called first");
        st.start.elapsed().as_micros() as u64
    }

    fn tid() -> u64 {
        TID.with(|t| *t)
    }

    /// Initialize the profiler and set output path.
    pub fn init<P: AsRef<Path>>(path: P) {
        let out_path = path.as_ref().to_path_buf();
        let _ = STATE.set(ProfilerState {
            start: Instant::now(),
            out_path,
            pid: 1,
            is_on: AtomicBool::new(true),
            events: Mutex::new(Vec::new()),
        });
    }

    /// Shut down the profiler and write the Chrome Trace JSON.
    pub fn shutdown() {
        if let Some(st) = STATE.get() {
            // Stop accepting new events (best-effort; spans already in-flight may still push).
            st.is_on.store(false, Ordering::Release);

            // Write file
            if let Err(e) = write_trace_file(st) {
                eprintln!("profiler::shutdown failed to write trace: {e}");
            }
        }
    }

    fn write_trace_file(st: &ProfilerState) -> std::io::Result<()> {
        // Snapshot events
        let events = {
            let mut guard = st.events.lock().unwrap();
            std::mem::take(&mut *guard)
        };

        if let Some(parent) = st.out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = File::create(&st.out_path)?;
        let mut w = BufWriter::new(f);

        write!(w, "{{\"traceEvents\":[")?;
        let mut first = true;
        for ev in events {
            if !first {
                write!(w, ",")?;
            }
            first = false;
            match ev {
                TraceEvent::Complete {
                    name,
                    ts_us,
                    dur_us,
                    pid,
                    tid,
                    args,
                } => {
                    write!(w, "{{\"name\":")?;
                    write_json_string(&mut w, &name)?;
                    write!(
                        w,
                        ",\"cat\":\"ecs\",\"ph\":\"X\",\"ts\":{},\"dur\":{},\"pid\":{},\"tid\":{}",
                        ts_us, dur_us, pid, tid
                    )?;
                    if !args.is_empty() {
                        write!(w, ",\"args\":{{")?;
                        let mut a_first = true;
                        for (k, v) in args {
                            if !a_first {
                                write!(w, ",")?;
                            }
                            a_first = false;
                            write_json_string(&mut w, &k)?;
                            write!(w, ":")?;
                            v.write_json(&mut w)?;
                        }
                        write!(w, "}}")?;
                    }
                    write!(w, "}}")?;
                }
                TraceEvent::ThreadName { ts_us, pid, tid, name } => {
                    write!(
                        w,
                        "{{\"name\":\"thread_name\",\"ph\":\"M\",\"ts\":{},\"pid\":{},\"tid\":{},\"args\":{{\"name\":",
                        ts_us, pid, tid
                    )?;
                    write_json_string(&mut w, &name)?;
                    write!(w, "}}}}")?;
                }
            }
        }
        write!(w, "]}}")?;
        w.flush()?;
        Ok(())
    }

    fn write_json_string<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
        write!(w, "\"")?;
        for ch in s.chars() {
            match ch {
                '"' => write!(w, "\\\"")?,
                '\\' => write!(w, "\\\\")?,
                '\n' => write!(w, "\\n")?,
                '\r' => write!(w, "\\r")?,
                '\t' => write!(w, "\\t")?,
                c if c.is_control() => write!(w, "\\u{:04x}", c as u32)?,
                c => write!(w, "{c}")?,
            }
        }
        write!(w, "\"")?;
        Ok(())
    }

    fn push_event(ev: TraceEvent) {
        let st = match STATE.get() {
            Some(s) => s,
            None => return,
        };
        if !st.is_on.load(Ordering::Acquire) {
            return;
        }
        let mut guard = st.events.lock().unwrap();
        guard.push(ev);
    }

    /// Assign a human-friendly thread name (shown in Perfetto/Chrome tracing).
    pub fn thread_name(name: impl Into<String>) {
        let st = match STATE.get() {
            Some(s) => s,
            None => return,
        };
        let ev = TraceEvent::ThreadName {
            ts_us: now_us(),
            pid: st.pid,
            tid: tid(),
            name: name.into(),
        };
        push_event(ev);
    }

    /// Add an argument to the *next* span created on this thread.
    pub fn next_arg(key: impl Into<String>, value: super::Arg) {
        let (k, v) = (key.into(), value.into_enabled());
        PENDING_ARGS.with(|p| p.borrow_mut().push((k, v)));
    }

    /// Create a profiling span.
    pub fn span(name: impl Into<super::SpanName>) -> SpanGuard {
        let st = match STATE.get() {
            Some(s) => s,
            None => return SpanGuard::disabled(),
        };
        if !st.is_on.load(Ordering::Acquire) {
            return SpanGuard::disabled();
        }

        let tid = tid();
        let ts0 = now_us();

        // Pull any args queued for this next span.
        let args = PENDING_ARGS.with(|p| std::mem::take(&mut *p.borrow_mut()));

        SpanGuard {
            name: name.into().0.into_owned(),
            ts0,
            tid,
            pid: st.pid,
            args,
            active: true,
        }
    }

    /// Create a profiling span using format_args without forcing the caller to allocate manually.
    pub fn span_fmt(args: fmt::Arguments<'_>) -> SpanGuard {
        span(args.to_string())
    }

    /// A RAII guard that records a Chrome Trace complete event on drop.
    pub struct SpanGuard {
        name: String,
        ts0: u64,
        tid: u64,
        pid: u32,
        args: Vec<(String, ArgValue)>,
        active: bool,
    }

    impl SpanGuard {
        fn disabled() -> Self {
            Self {
                name: String::new(),
                ts0: 0,
                tid: 0,
                pid: 0,
                args: Vec::new(),
                active: false,
            }
        }

        /// Attach an argument to this span (builder-style).
        #[inline]
        pub fn arg(mut self, key: impl Into<String>, value: super::Arg) -> Self {
            if self.active {
                self.args.push((key.into(), value.into_enabled()));
            }
            self
        }
    }

    impl Drop for SpanGuard {
        fn drop(&mut self) {
            if !self.active {
                return;
            }
            let ts1 = now_us();
            let dur = ts1.saturating_sub(self.ts0);
            push_event(TraceEvent::Complete {
                name: std::mem::take(&mut self.name),
                ts_us: self.ts0,
                dur_us: dur,
                pid: self.pid,
                tid: self.tid,
                args: std::mem::take(&mut self.args),
            });
        }
    }
}

#[cfg(not(feature = "profiling"))]
mod disabled {
    use super::*;

    /// Initialize profiler (no-op when profiling is disabled).
    #[inline]
    pub fn init<P: AsRef<Path>>(_path: P) {}

    /// Shut down profiler (no-op).
    #[inline]
    pub fn shutdown() {}

    /// Set thread name (no-op).
    #[inline]
    pub fn thread_name(_name: impl Into<String>) {}

    /// Attach arg to next span (no-op).
    #[inline]
    pub fn next_arg(_key: impl Into<String>, _value: super::Arg) {}

    /// Create profiling span (no-op).
    #[inline]
    pub fn span(_name: impl Into<super::SpanName>) -> SpanGuard {
        SpanGuard
    }

    /// Create profiling span using format_args (no-op).
    #[inline]
    pub fn span_fmt(_args: fmt::Arguments<'_>) -> SpanGuard {
        SpanGuard
    }

    /// No-op span guard.
    pub struct SpanGuard;

    impl SpanGuard {
        /// Attach an argument to this span (builder-style; no-op).
        #[inline]
        pub fn arg(self, _key: impl Into<String>, _value: super::Arg) -> Self {
            self
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API surface (stable regardless of feature flag)
// ─────────────────────────────────────────────────────────────────────────────

/// A span name; accepts `&'static str`, `String`, or `Cow<'static, str>`.
pub struct SpanName(pub Cow<'static, str>);

impl From<&'static str> for SpanName {
    fn from(s: &'static str) -> Self {
        SpanName(Cow::Borrowed(s))
    }
}
impl From<String> for SpanName {
    fn from(s: String) -> Self {
        SpanName(Cow::Owned(s))
    }
}
impl From<Cow<'static, str>> for SpanName {
    fn from(s: Cow<'static, str>) -> Self {
        SpanName(s)
    }
}

/// Argument value for profiling spans.
///
/// These values are serialized into the `args` field of Chrome Trace
/// events and can be inspected in Perfetto or `chrome://tracing`.
pub enum Arg {
    /// UTF-8 string value.
    Str(String),

    /// Unsigned 64-bit integer value.
    U64(u64),

    /// Signed 64-bit integer value.
    I64(i64),

    /// 64-bit floating-point value.
    F64(f64),
    
    /// Boolean value.
    Bool(bool),
}

// Re-export correct backend
#[cfg(feature = "profiling")]
pub use enabled::SpanGuard as SpanGuard;

#[cfg(not(feature = "profiling"))]
pub use disabled::SpanGuard as SpanGuard;

#[cfg(feature = "profiling")]
pub use enabled::{init, next_arg, shutdown, span, span_fmt, thread_name};

#[cfg(not(feature = "profiling"))]
pub use disabled::{init, next_arg, shutdown, span, span_fmt, thread_name};

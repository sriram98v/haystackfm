//! Software prefetch hint, used to hide LF-walk pointer-chasing latency in
//! `resolve_sa_batch` (see `fm_index/query.rs`). Independent LF-walk chains issue a
//! prefetch for next round's memory before consuming the current round's result, giving
//! the memory subsystem a head start while other lanes' arithmetic executes.
//!
//! No-op on targets without an intrinsic (e.g. non-x86_64, or x86_64 without SSE — which
//! doesn't happen in practice but is handled defensively): the independent-chain structure
//! of the batch loop still gives the CPU's out-of-order window a chance to overlap misses
//! even without an explicit hint.

/// Prefetch the cache line containing `ptr` into all cache levels (`T0` hint) ahead of a
/// read that's about to happen on another lane's next loop iteration.
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr, options(nostack, preserves_flags));
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "sse"),
        target_arch = "aarch64"
    )))]
    {
        let _ = ptr;
    }
}

# Per-android timing for the /revealed/ CG and FW runners.
#
# The CG runners (run_method_tracked in cpm.jl, run_ad_tracked in cpm_ad.jl,
# run_method_tracked_fw in frankwolfe/frankwolfe.jl) spend their wall-clock in
# two places: the SEPARATION oracle (one solve per android/class per iteration)
# and the WEALTH-REDISTRIBUTION master LP (re-solving the surrogate weights
# each iteration). This file accumulates both via two macros so call sites stay
# one-line:
#
#   cand = @time_sep class solve_separation_class(class, Ξ, u; ...)   # per-android
#   w, ... = @time_redist solve_wealth_redist_primal(Ξ, γ; ...)       # master
#
# Lifecycle: each runner calls `reset_cg_timers!()` on entry and
# `print_cg_timing_summary()` on exit. Accumulators are module-global; the
# AD pricing scan runs under `Threads.@threads`, so all mutation goes through
# `_TIMER_LOCK`.
#
# Must be included BEFORE separation.jl and the runners (macros need to exist at
# parse time). print_cg_timing_summary uses print_banner/print_config from
# logging.jl, but only at call time, so logging.jl may load after this file.

using Printf   # @sprintf in print_cg_timing_summary (independent of logging.jl's load order)

# Per-android separation seconds + call counts, keyed by class symbol
# (:ces, :linear, :leontief, :ql, :ges, :nn).
const _SEP_TIME = Dict{Symbol,Float64}()
const _SEP_CALLS = Dict{Symbol,Int}()
# Wealth-redistribution (master LP) seconds + call count.
const _REDIST_TIME = Ref(0.0)
const _REDIST_CALLS = Ref(0)
# Guards every accumulator mutation (AD's per-good scan is threaded).
const _TIMER_LOCK = ReentrantLock()

"""
    reset_cg_timers!()

Zero the separation / redistribution accumulators. Called at the start of every
`run_*_tracked` so each run (and each phase of a phased driver) reports its own
breakdown.
"""
function reset_cg_timers!()
    lock(_TIMER_LOCK) do
        empty!(_SEP_TIME)
        empty!(_SEP_CALLS)
        _REDIST_TIME[] = 0.0
        _REDIST_CALLS[] = 0
    end
end

_acc_sep!(class::Symbol, dt::Float64) = lock(_TIMER_LOCK) do
    _SEP_TIME[class] = get(_SEP_TIME, class, 0.0) + dt
    _SEP_CALLS[class] = get(_SEP_CALLS, class, 0) + 1
end

_acc_redist!(dt::Float64) = lock(_TIMER_LOCK) do
    _REDIST_TIME[] += dt
    _REDIST_CALLS[] += 1
end

"""
    @time_sep class expr

Evaluate `expr` (a per-class separation-oracle call), accumulate its elapsed
time against android `class`, and return `expr`'s value.
"""
macro time_sep(class, expr)
    quote
        local _t = time()
        local _v = $(esc(expr))
        _acc_sep!($(esc(class)), time() - _t)
        _v
    end
end

"""
    @time_redist expr

Evaluate `expr` (a wealth-redistribution master solve), accumulate its elapsed
time, and return `expr`'s value.
"""
macro time_redist(expr)
    quote
        local _t = time()
        local _v = $(esc(expr))
        _acc_redist!(time() - _t)
        _v
    end
end

"""
    print_cg_timing_summary(; io=stdout)

Print the per-android separation breakdown and the wealth-redistribution total
accumulated since the last `reset_cg_timers!()`. Always called on runner exit
(independent of verbosity). Percentages are relative to the sum of all timed
work (separation + redistribution); the FW runner has no redistribution, so
that line shows 0.
"""
function print_cg_timing_summary(; io::IO=stdout, width::Int=92)
    # Snapshot under the lock, then release it before printing.
    sep_by_class = lock(_TIMER_LOCK) do
        [(c, _SEP_TIME[c], get(_SEP_CALLS, c, 0)) for c in keys(_SEP_TIME)]
    end
    sort!(sep_by_class; by=x -> -x[2])   # descending by time
    sep_total = isempty(sep_by_class) ? 0.0 : sum(x -> x[2], sep_by_class)
    sep_calls = isempty(sep_by_class) ? 0 : sum(x -> x[3], sep_by_class)
    redist_total = _REDIST_TIME[]
    redist_calls = _REDIST_CALLS[]
    grand = sep_total + redist_total
    pct(x) = grand > 0 ? 100.0 * x / grand : 0.0
    avg(t, n) = n > 0 ? t / n : 0.0   # mean seconds per call

    println(io, "-"^width)   # divider rule on top of the block
    println(io, " timing breakdown (separation by android + wealth redistribution)")
    # Per-android separation rows.
    for (c, t, n) in sep_by_class
        print_config("sep :$(c)",
            @sprintf("%.4fs (n=%d, avg=%.4fs, %.1f%%)", t, n, avg(t, n), pct(t));
            io=io, indent=true)
    end
    # The separation subtotal only adds information when ≥2 androids
    # contributed; with a single android it equals the row above.
    if length(sep_by_class) > 1
        print_config("separation (total)",
            @sprintf("%.4fs (n=%d, avg=%.4fs, %.1f%%)",
                sep_total, sep_calls, avg(sep_total, sep_calls), pct(sep_total)); io=io)
    end
    print_config("wealth redistribution",
        @sprintf("%.4fs (n=%d, avg=%.4fs, %.1f%%)",
            redist_total, redist_calls, avg(redist_total, redist_calls), pct(redist_total));
        io=io)
    print_config("timed total", @sprintf("%.4fs", grand); io=io)
    return nothing
end

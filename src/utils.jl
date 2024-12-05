# -----------------------------------------------------------------------
# utilities functions
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using Printf, LaTeXStrings

# -----------------------------------------------------------------------
# copy fields from that to this
# - if field is an array, copy it; 
# - otherwise, simply set the same field (e.g. model, functions, etc.)
# -----------------------------------------------------------------------
function copy_fields(this, that)
    for field in fieldnames(typeof(that))
        if typeof(getfield(that, field)) <: AbstractArray
            setfield!(this, field, copy(getfield(that, field)))
        else
            setfield!(this, field, getfield(that, field))
        end
    end
end

# projection back (stepsize control)
proj(x) = x < 0 ? Inf : x

# -----------------------------------------------------------------------
# self-concordant barrier functions
# -----------------------------------------------------------------------
function logbar(x::AbstractArray)
    return -sum(log.(x))
end

function logbar(x::AbstractArray, w::AbstractVector)
    return -sum(w .* log.(x))
end

# --------------------------------------
# formatter
# --------------------------------------
const HEADER = [
    "ExchangeMarket.jl: A Julia Package for Exchange Market",
    "© Chuwen Zhang (2024)",
]

function format_header(log)
    loglength = log |> length
    sep = string(repeat("-", loglength))
    a = @sprintf("%s\n", sep)
    # print header lines in the middle
    for name in HEADER
        pref = loglength - (name |> length)
        prefs = string(repeat(" ", pref / 2 |> round |> Int))
        a *= @sprintf("%s%s%s\n", prefs, name, prefs)
    end
    return a * @sprintf("%s", sep), sep
end

mutable struct ExchangeLoggerUtil
    _logheadvals
    _logformats
    _dummy
    _dummyslots
    _loghead
    _blockheader
    _sep

    ExchangeLoggerUtil() = (
        this = new();
        this._logheadvals = ["k" "lg(μ)" "φ" "|∇φ|" "|Δp|" "t" "tₗ" "α" "kᵢ"];
        this._logformats = ["%7d |" " %+6.2f |" " %+10.4e |" " %.1e |" " %.1e |" " %.1e |" " %.1e |" " %.1e |" " %.1e "] .|> Printf.Format;
        this._dummy = [1 1.33e-12 1e-3 1e-3 1e2 1e4 1e4 1e4 1e4];
        this._dummyslots = map((y, ff) -> Printf.format(ff, y), this._dummy, this._logformats) .|> length;
        this._loghead = mapreduce((y, l) -> Printf.format(Printf.Format("%$(l-2)s |"), y), *, this._logheadvals, this._dummyslots)[1:end-1];
        (this._blockheader, this._sep) = format_header(this._loghead);
        return this
    )
end

produce_log(log, _logvals) = begin
    _logslots = map((y, ff) -> Printf.format(ff, y), _logvals, log._logformats)
    _logline = reduce(*, _logslots)
    return _logline
end

__default_logger = ExchangeLoggerUtil()
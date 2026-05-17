# Shared iteration-table logger for the /revealed/ CG and FW runners.
#
# Mirrors the ExchangeLoggerUtil pattern in src/utils.jl: column widths are
# pre-computed from a "dummy" row of canonical sample values, so the table
# header / separator are built once and the per-iter call site doesn't need
# to repeat hand-tuned %5/%10/%14 magic numbers.
#
# Used by cpm.jl (run_method_tracked) and frankwolfe.jl (run_fw_tracked /
# run_fw_FrankWolfe_tracked).

using Printf

# Shared banner title for all /revealed/ runners (cpm.jl, frankwolfe.jl,
# third-party/wrapper_frankwolfe.jl). Edit here to update everywhere.
const BANNER_TITLE = [
    "ExchangeMarket.jl: A Julia Package for Exchange Market",
    "Constructive Rationalization",
    "© Chuwen Zhang (2024)",
]

struct IterTable
    headers::Vector{String}
    formats::Vector{Printf.Format}
    widths::Vector{Int}
    sep::String
    header_line::String
end

"""
    IterTable(headers, formats, dummies)

Build a fixed-width iteration table from
- `headers` : column titles
- `formats` : Printf format strings, one per column (e.g. `"%10.3e"`)
- `dummies` : a row of representative values; each column's width is the
              max of the header length and the formatted dummy length, so
              both header and a typical row line up without truncation.
"""
function IterTable(headers::Vector{String}, formats::Vector{String}, dummies::Vector)
    @assert length(headers) == length(formats) == length(dummies)
    fs = [Printf.Format(f) for f in formats]
    widths = Int[max(length(h), length(Printf.format(f, d)))
                 for (h, f, d) in zip(headers, fs, dummies)]
    header_line = join((lpad(h, w) for (h, w) in zip(headers, widths)), " | ")
    sep         = join(("-"^w for w in widths), "-+-")
    return IterTable(headers, fs, widths, sep, header_line)
end

table_width(t::IterTable) = length(t.sep)

"""
    print_header(t; io)

Emit the column header and the separator beneath it.
"""
function print_header(t::IterTable; io::IO=stdout)
    println(io, t.header_line)
    println(io, t.sep)
end

"""
    print_row(t, vals; io)

Emit one row. `NaN` floats and `nothing` render as a left-padded dash so
optional metrics (test error not yet evaluated, reduced cost on a header
row) don't need ad-hoc handling at the call site.
"""
function print_row(t::IterTable, vals::AbstractVector; io::IO=stdout)
    @assert length(vals) == length(t.formats)
    out = Vector{String}(undef, length(vals))
    @inbounds for i in eachindex(vals)
        v = vals[i]
        s = if (v isa AbstractFloat && isnan(v)) || v === nothing
            "-"
        else
            Printf.format(t.formats[i], v)
        end
        out[i] = lpad(s, t.widths[i])
    end
    println(io, join(out, " | "))
end

"""
    print_continuation(t, msg; io)

Print a sub-event line beneath an iteration row, with `|- ` aligned to
the first inter-column `|` of the table. Used for between-iter messages
like "stage N stalled ...", "converged ...", "time limit reached ..." so
they read as children of the iteration they belong to.
"""
function print_continuation(t::IterTable, msg::AbstractString; io::IO=stdout)
    println(io, " "^(t.widths[1] + 1) * "|- " * msg)
end

"""
    print_banner(title_lines; width, io)
    print_banner(t::IterTable, title_lines; io)

Print the framed banner: separator + centered title lines + separator.
The `IterTable` form picks `width` from the table so the banner aligns
with the table beneath it.
"""
function print_banner(title_lines::Vector{String}; width::Int=92, io::IO=stdout)
    sep = "-"^width
    println(io, sep)
    for line in title_lines
        pad = max(0, div(width - length(line), 2))
        println(io, " "^pad * line)
    end
    println(io, sep)
end

print_banner(t::IterTable, title_lines::Vector{String}; io::IO=stdout) =
    print_banner(title_lines; width=table_width(t), io=io)

"""
    print_config(key, value; keywidth, indent, io)

Emit one config row of the form ` key := value` (or `  |- key := value`
when `indent=true`, used for per-class sub-blocks). `value` is stringified
via `string(...)`, so callers can pass numbers, symbols, or pre-formatted
`@sprintf` strings interchangeably.
"""
function print_config(key::AbstractString, value; keywidth::Int=30,
                      indent::Bool=false, io::IO=stdout)
    # Align the `:=` column across indented and non-indented rows: the
    # non-indent prefix is `" "` (1 char) and the indent prefix is
    # `"  |- "` (5 chars), so the indent key width must shrink by 4 to
    # keep the `:=` in the same column. (The original ad-hoc code in
    # cpm.jl / frankwolfe.jl used `-2` and was misaligned.)
    if indent
        @printf(io, "  |- %-*s := %s\n", keywidth - 4, key, string(value))
    else
        @printf(io,  " %-*s := %s\n", keywidth, key, string(value))
    end
end

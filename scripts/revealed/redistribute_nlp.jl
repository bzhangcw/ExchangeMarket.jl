# NLP wealth-redistribution master — heterogeneous-column variant.
#
# Used in place of the LP master in `redistribute.jl` whenever a non-
# homothetic separation class (currently `:ql`) is active and the column
# contribution `w_i · γ_i(p_k, w_i)` is not linear in `w_i`.
#
# Each entry of `columns` is either:
#   - an `AbstractMatrix{T}` of shape `(K, n)` — precomputed γ for a
#     linear-in-w_i column; contribution at (k, j) is `γ[k, j] · w_i`,
#   - a callback `(model, w_i, Ξ) -> Matrix(K, n)` that adds its own
#     auxiliaries / nonlinear expressions to the JuMP model and returns
#     the contribution as JuMP expressions.
#
# The master sums the per-column contributions into the standard balance
# constraint without inspecting them, so each android class owns its
# own JuMP encoding (see `make_ql_column` in `androids/ql.jl`).
#
# Return shape matches `solve_wealth_redist_primal` so the
# downstream `extract_duals(model, balance, budget, K, n)` and the
# separation-oracle interface need no changes.

using JuMP
using LinearAlgebra
using MadNLP

const ColumnSpec = Union{AbstractMatrix,Function}

"""
    solve_wealth_redist_nlp(Ξ, columns;
        verbose=false, timelimit=nothing, cache=nothing)

NLP variant of the wealth-redistribution primal, solved with MadNLP.
Objective is sum-of-squares residuals `Σ_{k,j} s[k,j]^2` (quadratic
loss), so no auxiliary L∞ lift is needed.

Arguments:
- `Ξ`           : `K` observations `(p_k, g_k)`.
- `columns`     : length-`m` vector. Each entry is either an
                  `AbstractMatrix{T}` of shape `(K, n)` (linear column,
                  contribution `γ[k,j] · w_i`) or a callback
                  `(model, w_i, Ξ) -> Matrix(K, n)` (nonlinear column,
                  installs its own auxiliaries).

Keywords:
- `verbose`     : if `false`, sets MadNLP `print_level = ERROR`.
- `timelimit`   : seconds; forwarded as MadNLP `max_wall_time`.
- `cache`       : accepted for signature parity with the LP master,
                  ignored in v1 (model is rebuilt every call).

Returns `(w, s, model, balance, budget)` — same shape as
`solve_wealth_redist_primal`.
"""
function solve_wealth_redist_nlp(
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    columns::Vector;
    verbose::Bool=false,
    timelimit::Union{Real,Nothing}=nothing,
    cache::Union{Ref,Nothing}=nothing,
) where T
    m = length(columns)
    K = length(Ξ)
    n = length(Ξ[1][1])

    model = new_model(nlp=true)
    if !verbose
        set_attribute(model, "print_level", MadNLP.ERROR)
    end
    if !isnothing(timelimit) && timelimit > 0
        set_attribute(model, "max_wall_time", Float64(timelimit))
    end

    @variable(model, w[1:m] >= 0)
    @variable(model, s[1:K, 1:n])          # signed residuals

    # Per-column contribution matrices (K × n). Element type is `Any`
    # because mixed-column models hold both `AffExpr` (from matrix
    # columns) and `NonlinearExpr` (from callback columns).
    contrib = Vector{Matrix{Any}}(undef, m)
    for i in 1:m
        c = columns[i]
        if c isa AbstractMatrix
            size(c) == (K, n) || error(
                "linear column $i must be ($K, $n), got $(size(c)).")
            contrib[i] = Matrix{Any}([c[k, j] * w[i] for k in 1:K, j in 1:n])
        else
            mat = c(model, w[i], Ξ)
            size(mat) == (K, n) || error(
                "callback column $i returned $(size(mat)), expected ($K, $n).")
            contrib[i] = Matrix{Any}(mat)
        end
    end

    # Per-observation balance: Σ_i contrib[i][k,j] + s[k,j] = p_k[j] g_k[j].
    balance = Matrix{ConstraintRef}(undef, K, n)
    for k in 1:K
        p_k, g_k = Ξ[k]
        Pg = p_k .* g_k
        for j in 1:n
            balance[k, j] = @constraint(model,
                sum(contrib[i][k, j] for i in 1:m) + s[k, j] == Pg[j])
        end
    end

    budget = @constraint(model, sum(w) == 1)
    @objective(model, Min, sum(s[k, j]^2 for k in 1:K, j in 1:n))

    optimize!(model)

    return value.(w), value.(s), model, balance, budget
end

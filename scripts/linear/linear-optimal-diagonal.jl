using JuMP, MosekTools

"""
    perturbed_optimal_diagonal(B; tolerance=1e-9)

Minimize the spectral condition number over all positive diagonal preconditioners:
    min t   subject to B <= Diagonal(d) <= t*B.

Jacobi equilibration followed by Cholesky whitening of the second LMI is an
invertible change of coordinates, not a restriction of admissible diagonals.
The SDP objective is divided by the Jacobi condition number to keep it O(1).
Return achieved conditioning and an independently repaired feasible dual bound.
`certified` means the relative primal/dual gap is at most 1e-5; inspect it even
if the solver reports OPTIMAL. MOSEK setup and its internal factorizations are
included in setup time, but are not counted as Newton matrix factorizations.
"""
function _perturbed_optimal_diagonal(B; tolerance=1e-9)
    n = size(B, 1)
    J = diag(B)
    all(>(0), J) || throw(ArgumentError("positive diagonal required"))
    A = Matrix(Symmetric(B ./ sqrt.(J * J')))
    factor = cholesky(Symmetric(A))
    reference = perturbed_condition(A)
    C = (factor.L \ Matrix{Float64}(I, n, n)) / sqrt(reference)
    model = Model(Mosek.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)
    for attribute in ("MSK_DPAR_INTPNT_CO_TOL_PFEAS", "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
                      "MSK_DPAR_INTPNT_CO_TOL_REL_GAP")
        set_optimizer_attribute(model, attribute, tolerance)
    end
    @variable(model, t)
    @variable(model, d[1:n])
    lower = @constraint(model, Symmetric(Diagonal(d) - A) in PSDCone())
    upper = @constraint(model, Symmetric(t*Matrix{Float64}(I,n,n) - C*Diagonal(d)*C') in PSDCone())
    @objective(model, Min, t)
    optimize!(model)
    status = termination_status(model)
    has_values(model) && has_duals(model) || error("optimal diagonal SDP failed: $status")
    dv, tv = value.(d), value(t)
    all(>(0), dv) || error("SDP returned a nonpositive diagonal")
    function psd_projection(M)
        E = eigen(Symmetric(M))
        E.vectors * Diagonal(max.(E.values, 0.0)) * E.vectors'
    end
    Y = psd_projection(Matrix(dual(lower)))
    Z = psd_projection(Matrix(dual(upper)))
    Z ./= tr(Z)
    target = max.(diag(C' * Z * C), 0.0)
    # PSD diagonal congruence enforces diag(Y)=diag(C'ZC) without
    # introducing a large error along the smallest eigenvector of B.
    Y += eps(Float64)*I
    scale = sqrt.(target ./ diag(Y))
    Y = (scale .* Y) .* scale'
    lower_bound = reference * dot(A, Y)
    diagonal = J .* dv
    achieved = perturbed_condition(B, diagonal)
    gap = max(0.0, achieved-lower_bound) / max(1.0, achieved)
    primal_violation = max(0.0, -eigmin(Symmetric(Diagonal(dv)-A)),
        -eigmin(Symmetric(tv*I-C*Diagonal(dv)*C')))
    dual_violation = max(norm(diag(Y)-diag(C'*Z*C), Inf), abs(tr(Z)-1),
        max(0.0,-eigmin(Symmetric(Y))), max(0.0,-eigmin(Symmetric(Z))))
    certified = gap <= max(1e-5, 100tolerance) &&
        dual_violation <= max(1e-7, 100tolerance) && lower_bound <= achieved*(1+1e-6)
    return (; diagonal, condition=achieved, lower_bound, relative_gap=gap, certified,
        objective=reference*tv, primal_violation, dual_violation, status,
        iterations=barrier_iterations(model), reference_condition=reference)
end

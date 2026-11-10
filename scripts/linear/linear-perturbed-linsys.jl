using LinearAlgebra, LinearOperators, Krylov, SparseArrays

"""Symmetric spectral condition number after diagonal congruence scaling."""
function perturbed_condition(B, diagonal=ones(size(B, 1)))
    all(>(0), diagonal) || throw(ArgumentError("preconditioner must be positive"))
    scale = 1 ./ sqrt.(diagonal)
    values = eigvals(Symmetric((scale .* B) .* scale'))
    return values[end] / values[1]
end

# Optional SDP diagnostics are loaded only when requested; benchmarks need no MOSEK.
function perturbed_optimal_diagonal(B; kwargs...)
    if !isdefined(@__MODULE__, :_perturbed_optimal_diagonal)
        Base.include(@__MODULE__, joinpath(@__DIR__, "linear-optimal-diagonal.jl"))
    end
    Base.invokelatest(() -> _perturbed_optimal_diagonal(B; kwargs...))
end

"""
Row-sum-preserving DR1 majorant of B = D - V*V'.
With t=V'1 and z=V*t/||t||, M=D-z*z' >= B and M*1=B*1.
Its Sherman-Morrison inverse is applied without factorization.
"""
function perturbed_dr1(B, V, row_sum)
    n = size(B, 1)
    t = vec(sum(V; dims=1))
    z = norm(t) == 0 ? zeros(n) : V * (t / norm(t))
    # D = row_sum + V*(V'1); use the row-sum identity to avoid
    # subtracting almost equal diagonal and rank-one contributions.
    diagonal = row_sum .+ z .* sum(z)
    a = z ./ diagonal
    # 1-z'D^-1z = sum(z .* row_sum ./ diagonal) / sum(z).
    denominator = sum(z) == 0 ? 1.0 : dot(z, row_sum ./ diagonal) / sum(z)
    denominator > 0 || error("nonpositive DR1 denominator")
    apply(r) = r ./ diagonal .+ a .* (dot(a, r) / denominator)
    return (; diagonal, z, denominator, apply)
end

"""
Solve B*d=b using Krylov's native centered preconditioning and stopping rule.
Minv applies the inverse preconditioner (ldiv=false); no centered matrix is
formed. With x0, solve B*correction=b-B*x0 and return x0+correction.
Krylov tests its preconditioned residual against rtol times its initial norm.
Report the final original residual separately, without using it to stop CG.
"""
function perturbed_pcg(B, b, Minv=I; x0=nothing,
    rtol=1e-9, maxiter=max(20, 5length(b)))
    offset = x0 === nothing ? zeros(length(b)) : Vector{Float64}(x0)
    length(offset) == length(b) || throw(ArgumentError("initial direction has wrong length"))
    all(isfinite, offset) || throw(ArgumentError("initial direction must be finite"))
    rhs = x0 === nothing ? b : b-B*offset
    correction, stats = Krylov.cg(B, rhs; M=Minv, ldiv=false,
        atol=0.0, rtol, itmax=maxiter, history=true)
    direction = offset + correction
    bnorm = norm(b)
    residual_norm = norm(B*direction-b)
    relative_residual = bnorm == 0 ? (residual_norm == 0 ? 0.0 : Inf) : residual_norm/bnorm
    initial_relative_residual = bnorm == 0 ? (norm(rhs)==0 ? 0.0 : Inf) : norm(rhs)/bnorm
    return (; direction, success=stats.solved && all(isfinite,direction),
        iterations=stats.niter, relative_residual, initial_relative_residual,
        weighted_residuals=copy(stats.residuals), backend_status=stats.status)
end

"""
    linear_perturbed_solve(B, b; method=:cholesky, row_sum=B*ones(n), V=nothing,
                          optimal_solver=nothing, initial_guess=:zero, rtol=1e-9, maxiter=5n)

Solve one identical Newton system by :cholesky, :cg, :pcg_jacobi,
:pcg_rowsum, :pcg_optimal, :pcg_dr1, :pcg_ic, or the approximate :dr1 direction.
`:pcg_ic` uses incomplete Cholesky without fill (`ic_drop=0`, IC(0)).
`ic_normalize=true` scalar-normalizes its inverse using the initial correction
rhs, avoiding scale-dependent machine-precision termination in native CG.
IC setup and normalization are included in setup_seconds; no fallback is used.
`optimal_solver(B)` must return a named tuple with a `diagonal` field.
No fallback is hidden. A direct DR1 step is explicitly marked approximate;
its original-system residual is reported even if it is large.
Krylov.cg(B, rhs; M=Minv, ldiv=false) handles centered preconditioning.
Minv is D^-1 for diagonal scaling and an analytic inverse operator for DR1.
The native stopping rule uses atol=0 and the requested rtol; the final
original residual is reported separately. No custom stopping callback,
residual-correction loop, dense whitener, or centered matrix is used.
`initial_guess=:dr1` starts an iterative solve at M_DR1^-1*b using one
Sherman-Morrison application; the CG preconditioner is unchanged. The default
`:zero` uses a zero starting direction. With a DR1 start, Krylov's relative
tolerance is measured against the initial preconditioned correction rhs.
No factorization is used for the guess.
The input B and b are never modified. SDP setup is included in setup_seconds.
"""
function linear_perturbed_solve(B, b; method=:cholesky,
    row_sum=vec(sum(B; dims=2)), V=nothing, optimal_solver=nothing, initial_guess=:zero,
    rtol=1e-9, maxiter=max(20, 5length(b)), ic_drop=0.0, ic_normalize::Bool=true, workspace=nothing)
    method in (:cholesky, :cg, :pcg_jacobi, :pcg_rowsum, :pcg_optimal, :pcg_dr1, :pcg_ic, :dr1) ||
        throw(ArgumentError("unknown linear solver: $method"))
    0 < rtol < 1 || throw(ArgumentError("linear tolerance must be between zero and one"))
    maxiter > 0 || throw(ArgumentError("linear iteration limit must be positive"))
    initial_guess in (:zero, :dr1) || throw(ArgumentError("unknown initial guess: $initial_guess"))
    iterative = method in (:cg, :pcg_jacobi, :pcg_rowsum, :pcg_optimal, :pcg_dr1, :pcg_ic)
    method != :pcg_ic || (isfinite(ic_drop) && ic_drop >= 0) ||
        throw(ArgumentError("IC drop must be finite and nonnegative"))
    initial_guess == :zero || iterative || throw(ArgumentError("initial guess applies only to iterative methods"))
    B = Symmetric(B)
    x0 = nothing
    factor = nothing
    info = nothing
    Minv = I
    setup_seconds = @elapsed begin
        if method == :cholesky
            factor = cholesky(B; check=false)
        elseif method == :pcg_jacobi
            diagonal = diag(B)
            Minv = Diagonal(1 ./ diagonal)
        elseif method == :pcg_rowsum
            all(>(0), row_sum) || throw(ArgumentError("row sums must be positive"))
            Minv = Diagonal(1 ./ row_sum)
        elseif method == :pcg_optimal
            optimal_solver === nothing && throw(ArgumentError("provide optimal_solver for :pcg_optimal"))
            info = optimal_solver(B)
            Minv = Diagonal(1 ./ info.diagonal)
        elseif method == :pcg_ic
            info = perturbed_ic(B; drop=ic_drop)
        elseif method in (:pcg_dr1, :dr1)
            V === nothing && throw(ArgumentError("DR1 requires the rank vectors V"))
            info = perturbed_dr1(B, V, row_sum)
            if method == :pcg_dr1
                Minv = LinearOperator(Float64, length(b), length(b), true, true,
                    (out, v) -> copyto!(out, info.apply(v)))
            end
        end
        if initial_guess == :dr1
            V === nothing && throw(ArgumentError("DR1 initial guess requires the rank vectors V"))
            initializer = method == :pcg_dr1 ? info : perturbed_dr1(B, V, row_sum)
            x0 = initializer.apply(b)
        end
        if method == :pcg_ic
            rhs = x0 === nothing ? b : b - B*x0
            normalized = ic_normalize ? perturbed_scaled_inverse(info.apply, rhs) : nothing
            inverse_apply = ic_normalize ? normalized.apply : info.apply
            info = merge(info, (normalized=ic_normalize,
                inverse_scale=ic_normalize ? normalized.scale : 1.0,
                normalization_applications=ic_normalize ? normalized.setup_applications : 0))
            Minv = LinearOperator(Float64, length(b), length(b), true, true,
                (out, v) -> copyto!(out, inverse_apply(v)))
        end
    end
    result = nothing
    solve_seconds = @elapsed begin
        if method == :cholesky
            ok = issuccess(factor)
            direction = ok ? factor \ b : zeros(length(b))
            residual = norm(B*direction-b) / max(norm(b), eps(Float64))
            result = (; direction, success=ok && residual <= rtol, iterations=0,
                residuals=[residual])
        elseif method == :dr1
            direction = info.apply(b)
            residual = norm(B*direction-b) / max(norm(b), eps(Float64))
            result = (; direction, success=all(isfinite, direction), iterations=0,
                residuals=[residual])
        else
            result = perturbed_pcg(B, b, Minv; x0, rtol, maxiter)
        end
    end
    return merge(result, (; method, approximate=method == :dr1, initial_guess,
        initializer_applications=initial_guess == :dr1 ? 1 : 0,
        initial_relative_residual=iterative ? result.initial_relative_residual : NaN,
        factorizations=method == :cholesky ? 1 : 0,
        preconditioner_factorizations=method in (:pcg_optimal, :pcg_ic) ? 1 : 0,
        sdp_iterations=method == :pcg_optimal ? info.iterations : 0,
        setup_seconds, solve_seconds,
        relative_residual=iterative ? result.relative_residual : last(result.residuals), info,
        preconditioner=Minv))
end

"""Inverse symmetric Gauss-Seidel preconditioner: (D+L)'⁻¹ D (D+L)⁻¹."""
function perturbed_sgs(B)
    T = LowerTriangular(sparse(tril(Matrix(B))))
    diagonal = diag(B)
    all(>(0), diagonal) || throw(ArgumentError("positive diagonal required"))
    apply(r) = T' \ (diagonal .* (T \ r))
    return (; apply, factor=T, factor_nnz=nnz(parent(T)), factor_order=size(B,1))
end

"""
Incomplete Cholesky with no fill beyond a thresholded original pattern.
Retain B[i,j] when |B[i,j]| >= drop*sqrt(B[i,i]*B[j,j]); drop=0 is IC(0).
No diagonal shift or fallback is hidden. The dense setup loop is a reference
implementation; the returned factor and triangular applications are sparse.
"""
function perturbed_ic(B; drop=0.0)
    drop >= 0 || throw(ArgumentError("drop must be nonnegative"))
    n = size(B,1)
    L = zeros(n,n)
    diagonal = diag(B)
    all(>(0), diagonal) || throw(ArgumentError("positive diagonal required"))
    for j in 1:n
        pivot = B[j,j] - sum(abs2, view(L,j,1:j-1))
        pivot > 0 || error("incomplete Cholesky failed at pivot $j; no shift applied")
        L[j,j] = sqrt(pivot)
        for i in j+1:n
            if !iszero(B[i,j]) && abs(B[i,j]) >= drop*sqrt(diagonal[i]*diagonal[j])
                L[i,j] = (B[i,j] - dot(view(L,i,1:j-1),view(L,j,1:j-1))) / L[j,j]
            end
        end
    end
    factor = LowerTriangular(sparse(L))
    apply(r) = factor' \ (factor \ r)
    return (; apply, factor, factor_nnz=nnz(parent(factor)), factor_order=n, drop)
end

"""
Deterministic Lloyd groups of normalized buyer columns V[:,i]/sum(V[:,i]).
Farthest-point seeds and a fixed number of iterations avoid a random benchmark.
Zero columns contribute no rank term; empty groups are omitted by DRq.
"""
function perturbed_buyer_groups(V, q; iterations=8)
    n,m=size(V)
    1 <= q <= m || throw(ArgumentError("require 1 <= q <= number of buyers"))
    mass=vec(sum(V;dims=1))
    Y=V ./ max.(mass',eps(Float64))
    norms=vec(sum(abs2,Y;dims=1))
    centers=zeros(n,q)
    distance=fill(Inf,m)
    index=argmax(mass)
    for k in 1:q
        centers[:,k]=Y[:,index]
        distance=min.(distance, max.(0.0,norms .+ dot(centers[:,k],centers[:,k]) .- 2 .* (Y'*centers[:,k])))
        index=argmax(distance)
    end
    groups=ones(Int,m)
    for _ in 1:iterations
        distances=reshape(norms,1,m) .+ sum(abs2,centers;dims=1)' .- 2 .* (centers'*Y)
        groups=[argmin(view(distances,:,i)) for i in 1:m]
        for k in 1:q
            idx=findall(==(k),groups)
            isempty(idx) || (centers[:,k]=vec(sum(view(Y,:,idx);dims=2))/length(idx))
        end
    end
    return groups
end

"""
Row-sum-preserving grouped DRq majorant, extending perturbed_dr1.
For each disjoint group G, z_G=V_G*(V_G'1)/||V_G'1||.
M=D0-Z*Z' >= B and M*1=B*1. Apply its inverse with a q-by-q Cholesky.
The small Schur diagonal is assembled through its positive row-sum identity
in scaled coordinates to avoid cancellation in I-Z'D0^-1 Z.
"""
function perturbed_drq(B,V,row_sum; q=8, groups=nothing)
    n,m=size(V)
    length(row_sum)==n || throw(DimensionMismatch("row sums"))
    all(>(0),row_sum) || throw(ArgumentError("positive row sums required"))
    groups===nothing && (groups=perturbed_buyer_groups(V,q))
    length(groups)==m || throw(DimensionMismatch("buyer groups"))
    columns=Vector{Float64}[]
    masses=Float64[]
    for group in sort(unique(groups))
        W=V[:,findall(==(group),groups)]
        t=vec(sum(W;dims=1))
        weight=norm(t)
        weight==0 && continue
        push!(columns,W*(t/weight))
        push!(masses,weight)
    end
    Z=isempty(columns) ? zeros(n,0) : hcat(columns...)
    rank=size(Z,2)
    # Z*(Z'1)=V*(V'1), including for arbitrary disjoint buyer groups.
    diagonal=row_sum+Z*vec(sum(Z;dims=1))
    A=Z ./ diagonal
    S=-(Z'*A)
    if rank>0
        scaled_rows=Z'*(row_sum ./ diagonal)
        for i in 1:rank
            S[i,i]=(scaled_rows[i]-sum(S[i,j]*masses[j] for j in 1:rank if j!=i;init=0.0))/masses[i]
        end
    end
    factor=cholesky(Symmetric(S))
    apply(r)=r./diagonal + A*(factor\(A'*r))
    return (;apply,Z,diagonal,schur=Matrix(Symmetric(S)),factor,groups,rank,
        factor_order=rank,factor_nnz=rank*(rank+1)÷2)
end

"""
Buyer-augmented SPD system A=[D0 -V; -V' I], whose Schur complement is B.
`mode=:diagonal` uses diag(D0,I). `mode=:factor` uses
L*diag(M,I)*L', L=[I -V;0 I], and the supplied inverse M via price_apply.
Both actions are matrix-free. `evaluation=:factored` evaluates the exactly
same augmented matrix as L*diag(B,I)*L' using the stable stored B; the
`:blocks` action uses D0 and V directly and exposes cancellation effects.
Lift a price start to (d0,V'd0). Report the final original price residual,
augmented residual, and auxiliary-equation residual separately.
"""
function perturbed_augmented_solve(B,b,V; mode=:diagonal,price_apply=nothing,
    x0=nothing,rtol=1e-9,maxiter=max(20,5length(b)),evaluation=:factored)
    mode in (:diagonal,:factor) || throw(ArgumentError("unknown augmented preconditioner"))
    evaluation in (:factored,:blocks) || throw(ArgumentError("unknown augmented matvec"))
    n,m=size(V)
    size(B)==(n,n) && length(b)==n || throw(DimensionMismatch("augmented system"))
    mode==:factor && price_apply===nothing && throw(ArgumentError("factor mode requires price_apply"))
    diagonal=diag(B)+vec(sum(abs2,V;dims=2))
    function applyA!(out,u)
        d=view(u,1:n);y=view(u,n+1:n+m)
        t=V'*d-y
        if evaluation==:factored
            out[1:n]=B*d+V*t
        else
            out[1:n]=diagonal.*d-V*y
        end
        out[n+1:n+m]=-t
        out
    end
    A=LinearOperator(Float64,n+m,n+m,true,true,applyA!)
    function applyM!(out,r)
        first=view(r,1:n);last=view(r,n+1:n+m)
        if mode==:diagonal
            out[1:n]=first./diagonal
            out[n+1:n+m]=last
        else
            z=price_apply(first+V*last)
            out[1:n]=z
            out[n+1:n+m]=last+V'*z
        end
        out
    end
    Minv=LinearOperator(Float64,n+m,n+m,true,true,applyM!)
    offset=x0===nothing ? zeros(n) : Vector{Float64}(x0)
    # The lifted start satisfies the auxiliary equation exactly.
    rhs=vcat(b-B*offset,zeros(m))
    correction,stats=Krylov.cg(A,rhs;M=Minv,ldiv=false,atol=0.0,rtol,itmax=maxiter,history=true)
    direction=offset+correction[1:n]
    auxiliary=V'*offset+correction[n+1:n+m]
    original=B*direction-b
    mismatch=auxiliary-V'*direction
    augmented_first=original-V*mismatch
    ratio(v)=norm(b)==0 ? (norm(v)==0 ? 0.0 : Inf) : norm(v)/norm(b)
    return (;direction,auxiliary,success=stats.solved&&all(isfinite,direction),iterations=stats.niter,
        relative_residual=ratio(original),augmented_relative_residual=ratio(vcat(augmented_first,mismatch)),
        auxiliary_relative_residual=ratio(mismatch),initial_relative_residual=ratio(rhs),
        weighted_residuals=copy(stats.residuals),backend_status=stats.status,dimension=n+m,
        mode,evaluation,operator=A,preconditioner=Minv)
end

"""
Scalar normalization of an inverse preconditioner for a given initial residual.
Return alpha*M^-1 with alpha=1/(rhs'M^-1 rhs), so its initial weighted norm is
one. This changes neither the spectral condition number nor exact-arithmetic
relative-tolerance PCG iterates. It avoids scale-dependent early termination
by Krylov's absolute machine-precision residual guard. No rhs, matrix,
stopping rule, or residual-correction loop is modified. Count one setup
inverse application; pass the actual correction rhs when using a start.
"""
function perturbed_scaled_inverse(apply, rhs)
    weighted_squared=dot(rhs,apply(rhs))
    if iszero(norm(rhs))
        scale=1.0
    else
        isfinite(weighted_squared) && weighted_squared>0 ||
            throw(ArgumentError("inverse preconditioner must be positive definite on rhs"))
        scale=inv(weighted_squared)
        isfinite(scale) || throw(ArgumentError("preconditioner normalization overflow"))
    end
    scaled_apply(r)=scale .* apply(r)
    return (;apply=scaled_apply,scale,initial_weighted_norm=sqrt(weighted_squared),setup_applications=1)
end

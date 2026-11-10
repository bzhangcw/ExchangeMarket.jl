# Optional CUDA backend. Include linear-perturbed.jl and propres.jl first.
using CUDA, CUDSS
const LinearCUSPARSE = CUDA.CUSPARSE
CUDA.allowscalar(false)

linear_device(::CUDA.CuArray) = :gpu
linear_sync(::CUDA.CuArray) = CUDA.synchronize()
linear_default_solver(::CUDA.CuArray) = :cudss
linear_supports(::CUDA.CuArray) = nothing
linear_arrays(c::CUDA.CuArray, w::CUDA.CuArray) = throw(ArgumentError("GPU inputs must both be Float64 CUDA arrays"))
linear_arrays(c::CUDA.CuArray, w) = throw(ArgumentError("GPU inputs must both be resident Float64 CUDA arrays"))
linear_arrays(c::AbstractArray, w::CUDA.CuArray) = throw(ArgumentError("mixed CPU/GPU market arrays"))
function linear_arrays(c::CUDA.CuMatrix{Float64}, w::CUDA.CuVector{Float64})
    c, w                         # inputs stay resident and read-only
end
function linear_validate_solver(c::CUDA.CuArray, method)
    method in (:cudss, :pcg_ic) || throw(ArgumentError("GPU solver must be :cudss or :pcg_ic"))
end
function linear_prices(c::CUDA.CuArray, w, p::AbstractVector)
    p isa CUDA.CuVector{Float64} || throw(ArgumentError("GPU initial prices must already be a Float64 CuVector"))
    copy(p)
end
function linear_response_workspace(c::CUDA.CuArray, w)
    ws = linear_array_response_workspace(c, w)
    merge(ws, (rows=CUDA.CuArray(ws.rows),))
end
linear_metrics_workspace(c::CUDA.CuArray) = linear_array_metrics_workspace(c)
linear_pr_workspace(c::CUDA.CuArray, p) = linear_array_metrics_workspace(c)
linear_allocations(c::CUDA.CuArray, T, n, m) = similar(c, T, n, m)
function linear_equilibrium_metrics(c::CUDA.CuArray, w, p, x; demand=nothing, workspace=nothing)
    ws = workspace === nothing ? linear_metrics_workspace(c) : workspace
    linear_array_metrics(c, w, p, x, ws; demand=demand === nothing ? ws.demand : demand)
end
function linear_perturbed_response!(x::CUDA.CuArray, s, lam, delta, c, w, p, sigma;
    rtol=1e-12, maxiter=100, supports=nothing, workspace=nothing)
    ws = workspace === nothing ? linear_response_workspace(c, w) : workspace
    linear_array_response!(x, s, lam, delta, c, w, p, sigma, ws; rtol, maxiter)
end
function linear_perturbed_hessian!(H::CUDA.CuArray, V, diagonal, c, w, p, x, sigma;
    sensitivity=nothing, supports=nothing, workspace=nothing)
    ws = workspace === nothing ? linear_response_workspace(c, w) : workspace
    linear_array_hessian!(H, V, diagonal, c, w, p, x, sigma, ws; sensitivity)
end
function linear_set_diagonal!(H::CUDA.CuArray, diagonal)
    view(vec(H), 1:size(H,1)+1:length(H)) .= diagonal
    H
end
function linear_shift_diagonal!(H::CUDA.CuArray, shift)
    view(vec(H), 1:size(H,1)+1:length(H)) .+= shift
    H
end
pr_step!(x::CUDA.CuArray, p, c, w, ws) = linear_array_pr_step!(x, p, c, w, ws)

# This small indexing kernel caches the CSR-to-dense map once. All numerical
# assembly, factorization, and inverse applications remain on the GPU.
function linear_csr_map_kernel!(indices, rowptr, colval, n)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= n
        for k in rowptr[row]:rowptr[row+1]-1
            indices[k] = row + (colval[k]-1) * n
        end
    end
    return
end

function linear_gpu_pattern(c)
    n = size(c, 1)
    support = Float64.(c .> 0)
    pattern = support * support'
    # Preserve structural zeros even when numerical values later underflow.
    rows = reshape(CUDA.CuArray(Int32.(1:n)), :, 1)
    pattern .= Float64.((pattern .> 0) .& (rows .>= rows'))
    # Legacy cuSPARSE IC(0) requires sorted column indices within each row.
    A = LinearCUSPARSE.CuSparseMatrixCSR(pattern; sorted=true)
    indices = similar(A.colVal)
    CUDA.@cuda threads=256 blocks=cld(n,256) linear_csr_map_kernel!(indices, A.rowPtr, A.colVal, n)
    A, indices
end

# A persistent cuSPARSE SpSV plan: analysis/workspace are reused throughout CG.
# Descriptors own references to fixed scratch vectors, which avoids rebuilding
# sparse analysis for every preconditioner application.
struct LinearSpSVPlan{A,X,Y,D,B}
    matrix::A
    input::X
    output::Y
    descriptor::D
    buffer::B
    operation::Char
end
function linear_spsv_plan(A, x, y, operation)
    S = LinearCUSPARSE
    matrix = S.CuSparseMatrixDescriptor(A, 'O')
    S.cusparseSpMatSetAttribute(matrix, 'F', Ref{S.cusparseFillMode_t}('L'), Csize_t(sizeof(S.cusparseFillMode_t)))
    S.cusparseSpMatSetAttribute(matrix, 'D', Ref{S.cusparseDiagType_t}('N'), Csize_t(sizeof(S.cusparseDiagType_t)))
    input, output = S.CuDenseVectorDescriptor(x), S.CuDenseVectorDescriptor(y)
    descriptor = S.CuSparseSpSVDescriptor()
    algo = S.CUSPARSE_SPSV_ALG_DEFAULT
    bytes = Ref{Csize_t}()
    S.cusparseSpSV_bufferSize(S.handle(), operation, Ref(1.0), matrix, input, output,
        Float64, algo, descriptor, bytes)
    buffer = CUDA.CuVector{UInt8}(undef, bytes[])
    S.cusparseSpSV_analysis(S.handle(), operation, Ref(1.0), matrix, input, output,
        Float64, algo, descriptor, buffer)
    LinearSpSVPlan(matrix, input, output, descriptor, buffer, operation)
end
function linear_spsv_solve!(plan::LinearSpSVPlan)
    S = LinearCUSPARSE
    S.cusparseSpSV_solve(S.handle(), plan.operation, Ref(1.0), plan.matrix,
        plan.input, plan.output, Float64, S.CUSPARSE_SPSV_ALG_DEFAULT, plan.descriptor)
    nothing
end

mutable struct LinearGPUSolver{A,I,V,K}
    method::Symbol
    matrix::A
    indices::I
    direction::V
    residual::V
    rhs::V
    temporary::V
    inverse_result::V
    cg::K
    direct::Any
    forward::Any
    backward::Any
    inverse_scale::Float64
    analyzed::Bool
end
function linear_solver_workspace(c::CUDA.CuArray, w, method)
    linear_validate_solver(c, method)
    n = size(c, 1)
    A, indices = linear_gpu_pattern(c)
    vec() = CUDA.zeros(Float64, n)
    direction = vec()
    cg_type = isdefined(Krylov, :CgWorkspace) ? Krylov.CgWorkspace : Krylov.CgSolver
    cg = method == :pcg_ic ? cg_type(n, n, typeof(direction)) : nothing
    direct = method == :cudss ? CUDSS.CudssSolver(A, "SPD", 'L') : nothing
    LinearGPUSolver(method, A, indices, direction, vec(), vec(), vec(), vec(),
        cg, direct, nothing, nothing, 1.0, false)
end
function linear_gpu_inverse!(out, v, ws)
    copyto!(ws.rhs, v)
    linear_spsv_solve!(ws.forward)
    linear_spsv_solve!(ws.backward)
    out .= ws.inverse_scale .* ws.inverse_result
    nothing
end

function linear_perturbed_solve(B::CUDA.CuMatrix{Float64}, b::CUDA.CuVector{Float64};
    method=:cudss, row_sum=nothing, V=nothing, optimal_solver=nothing,
    initial_guess=:zero, rtol=1e-9, maxiter=max(20,5length(b)),
    ic_drop=0.0, ic_normalize::Bool=true, workspace=nothing)
    linear_validate_solver(B, method)
    initial_guess == :zero || throw(ArgumentError("GPU comparison uses zero initial directions"))
    ic_drop == 0 || throw(ArgumentError("native GPU IC backend currently implements IC(0)"))
    0 < rtol < 1 && maxiter > 0 || throw(ArgumentError("invalid linear solve tolerance or iteration limit"))
    workspace === nothing && throw(ArgumentError("construct a GPU solver workspace before iteration"))
    ws = workspace
    ws.method == method || throw(ArgumentError("workspace/solver mismatch"))
    size(ws.matrix) == size(B) || throw(DimensionMismatch("workspace/matrix mismatch"))
    CUDA.synchronize()
    setup_seconds = @elapsed begin
        ws.matrix.nzVal .= view(vec(B), ws.indices)
        if method == :cudss
            if !ws.analyzed
                CUDSS.cudss("analysis", ws.direct, ws.direction, b)
                ws.analyzed = true
            end
        else
            # Native IC(0) overwrites this separate matrix, never B.
            LinearCUSPARSE.ic02!(ws.matrix)
            # Numeric values changed: redo SpSV analysis once per Newton system.
            ws.forward = linear_spsv_plan(ws.matrix, ws.rhs, ws.temporary, 'N')
            ws.backward = linear_spsv_plan(ws.matrix, ws.temporary, ws.inverse_result, 'T')
            ws.inverse_scale = 1.0
            if ic_normalize
                linear_gpu_inverse!(ws.direction, b, ws)
                weighted = dot(b, ws.direction)
                if !iszero(norm(b))
                    isfinite(weighted) && weighted > 0 || error("IC inverse is not positive definite")
                    ws.inverse_scale = inv(weighted)
                    isfinite(ws.inverse_scale) || error("IC normalization overflow")
                end
            end
        end
        CUDA.synchronize()
    end
    iterations, success = 0, false
    solve_seconds = @elapsed begin
        if method == :cudss
            CUDSS.cudss("factorization", ws.direct, ws.direction, b)
            CUDA.synchronize()
            CUDSS.cudss_get(ws.direct, "info") == 0 || error("cuDSS SPD factorization failed")
            CUDSS.cudss("solve", ws.direct, ws.direction, b)
            success = true
        else
            Minv = LinearOperator(Float64, length(b), length(b), true, true,
                (out,v) -> linear_gpu_inverse!(out,v,ws); S=typeof(b))
            Krylov.cg!(ws.cg, B, b; M=Minv, ldiv=false, atol=0.0, rtol,
                itmax=maxiter, history=false)
            copyto!(ws.direction, ws.cg.x)
            iterations, success = ws.cg.stats.niter, ws.cg.stats.solved
        end
        CUDA.synchronize()
    end
    mul!(ws.residual, B, ws.direction)
    ws.residual .-= b
    bnorm, residual_norm = norm(b), norm(ws.residual)
    relative_residual = bnorm == 0 ? (residual_norm == 0 ? 0.0 : Inf) : residual_norm / bnorm
    success &= all(isfinite, ws.direction)
    method == :cudss && (success &= relative_residual <= rtol)
    (; direction=ws.direction, success, iterations, relative_residual,
       initial_relative_residual=bnorm == 0 ? 0.0 : 1.0,
       method, approximate=false, initial_guess, initializer_applications=0,
       factorizations=method == :cudss ? 1 : 0,
       preconditioner_factorizations=method == :pcg_ic ? 1 : 0,
       sdp_iterations=0, setup_seconds, solve_seconds)
end

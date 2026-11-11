# CUDA dispatch for the generic sparse linear-market representation.
const LinearSparseGPUArray = LinearSparseArray{P,V} where {P,V<:CUDA.CuVector}
function linear_to_device(a::LinearSparseArray,::Val{:gpu})
    p=a.pattern
    pattern=LinearSparsePattern(p.n,p.m,CuArray(p.colptr),CuArray(p.rowval),CuArray(p.rowptr),
        CuArray(p.colval),CuArray(p.csc_to_csr),CuArray(p.csr_to_csc))
    LinearSparseArray(pattern,CuArray(a.values),CuArray(a.base_rows),CuArray(a.base_cols),a.zero_spending)
end
linear_to_device(a::Array,::Val{:gpu})=CuArray(a)
function linear_sparse_gpu_kernel!(kernel::F,n,args::Vararg{Any,N}) where {F,N}
    index=(blockIdx().x-1)*blockDim().x+threadIdx().x
    index<=n && kernel(index,args...,n)
    return
end
function linear_sparse_launch(::CUDA.CuVector,kernel::F,n,args::Vararg{Any,N}) where {F,N}
    CUDA.@cuda threads=256 blocks=cld(n,256) linear_sparse_gpu_kernel!(kernel,n,args...)
    nothing
end
linear_validate_solver(c::LinearCUSPARSE.CuSparseMatrixCSR,method)=
    method in (:cudss,:pcg_ic) ? nothing : throw(ArgumentError("invalid sparse GPU solver"))
linear_sparse_rank_values(V::LinearSparseRankFactors{<:LinearCUSPARSE.CuSparseMatrixCSR})=(nonzeros(V.matrix),nonzeros(V.transposed))
function linear_sparse_product!(H::LinearSparseHessian{<:LinearCUSPARSE.CuSparseMatrixCSR},V)
    LinearCUSPARSE.gemm!('N','N',-1.0,V.matrix,V.transposed,0.0,H.matrix,'O')
    H
end
function linear_solver_workspace(c::LinearSparseGPUArray,w,method)
    linear_validate_solver(c,method)
    p=c.pattern;n,m=size(c)
    values=CUDA.ones(Float64,length(c.values))
    V=LinearSparseRankFactors(LinearCUSPARSE.CuSparseMatrixCSR(p.rowptr,p.colval,copy(values),(n,m)),
        LinearCUSPARSE.CuSparseMatrixCSR(p.colptr,p.rowval,values,(m,n)))
    matrix=LinearCUSPARSE.gemm('N','N',1.0,V.matrix,V.transposed,'O')
    diagonal_indices=CUDA.zeros(Int32,n)
    linear_sparse_launch(linear_sparse_diagonal_kernel!,n,diagonal_indices,matrix.rowPtr,matrix.colVal)
    minimum(diagonal_indices)>0 || error("sparse Hessian has a missing diagonal")
    H=LinearSparseHessian(matrix,diagonal_indices)
    counts=CUDA.zeros(Int32,n)
    linear_sparse_launch(linear_sparse_lower_counts_kernel!,n,counts,matrix.rowPtr,matrix.colVal)
    pointers=CUDA.ones(Int32,n+1)
    view(pointers,2:n+1) .= Int32(1) .+ cumsum(counts)
    count=Int(sum(Int64,counts))
    columns=CUDA.zeros(Int32,count);indices=similar(columns)
    linear_sparse_launch(linear_sparse_lower_fill_kernel!,n,columns,indices,pointers,matrix.rowPtr,matrix.colVal)
    A=LinearCUSPARSE.CuSparseMatrixCSR(pointers,columns,CUDA.zeros(Float64,count),(n,n))
    vec()=CUDA.zeros(Float64,n)
    direction=vec();cg_type=isdefined(Krylov,:CgWorkspace) ? Krylov.CgWorkspace : Krylov.CgSolver
    cg=method==:pcg_ic ? cg_type(n,n,typeof(direction)) : nothing
    direct=nothing # Avoid an unused, unanalyzed native cuDSS object.
    core=LinearGPUSolver(method,A,indices,direction,vec(),vec(),vec(),vec(),cg,direct,nothing,nothing,1.0,false)
    LinearSparseSolverWorkspace(core,H,V)
end
linear_gpu_update_values!(ws,B::LinearCUSPARSE.CuSparseMatrixCSR) =
    (ws.matrix.nzVal .= view(B.nzVal,ws.indices))
function linear_perturbed_solve(H::LinearSparseHessian{<:LinearCUSPARSE.CuSparseMatrixCSR},b;workspace,kwargs...)
    linear_gpu_solve(H.matrix,b;workspace=workspace.solver,kwargs...)
end

function linear_arrays(c::LinearSparseArray,w::CUDA.CuVector{Float64})
    linear_device(c)==:gpu || throw(ArgumentError("mixed CPU/GPU market arrays"))
    c,w
end

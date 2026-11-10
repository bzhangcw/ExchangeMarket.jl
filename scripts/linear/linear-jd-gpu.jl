# Sparse CUDA storage for the existing PR/IPM algorithms. Valuations and all
# numerical iterates stay on device. Zero-valued perturbed allocations are
# represented analytically, so the allocation is mathematically dense without
# allocating goods × buyers storage.
struct JDSparsePattern{I}
    n::Int
    m::Int
    colptr::I
    rowval::I
    rowptr::I
    colval::I
    csc_to_csr::I
    csr_to_csc::I
end
mutable struct JDSparseArray{P,V} <: AbstractMatrix{Float64}
    pattern::P
    values::V
    base_rows::V
    base_cols::V
    zero_spending::Float64 # p[j]*base_rows[j], constant for our implicit states
end
Base.size(a::JDSparseArray) = (a.pattern.n,a.pattern.m)
function Base.similar(a::JDSparseArray)
    JDSparseArray(a.pattern,CUDA.zeros(Float64,length(a.values)),
        CUDA.zeros(Float64,size(a,1)),CUDA.ones(Float64,size(a,2)),0.0)
end
function Base.copy(a::JDSparseArray)
    b=similar(a);linear_copy!(b,a);b
end
function linear_copy!(a::JDSparseArray,b::JDSparseArray)
    a.pattern===b.pattern || throw(ArgumentError("incompatible sparse allocation patterns"))
    copyto!(a.values,b.values);copyto!(a.base_rows,b.base_rows);copyto!(a.base_cols,b.base_cols)
    a.zero_spending=b.zero_spending
    a
end
function jd_gpu_array(c::SparseMatrixCSC{Float64})
    n,m=size(c);count=nnz(c)
    max(n,m,count+1)<typemax(Int32) || error("JD sparse GPU indices exceed Int32 capacity")
    rowcounts=zeros(Int32,n)
    for j in c.rowval;rowcounts[j]+=1;end
    rowptr=vcat(Int32(1),Int32(1).+cumsum(rowcounts))
    next=copy(rowptr[1:n]);colval=Vector{Int32}(undef,count)
    csc_to_csr=similar(colval);csr_to_csc=similar(colval)
    for i in 1:m, k in c.colptr[i]:c.colptr[i+1]-1
        j=c.rowval[k];q=next[j];next[j]+=1
        colval[q]=i;csc_to_csr[k]=q;csr_to_csc[q]=k
    end
    pattern=JDSparsePattern(n,m,CuArray(Int32.(c.colptr)),CuArray(Int32.(c.rowval)),
        CuArray(rowptr),CuArray(colval),CuArray(csc_to_csr),CuArray(csr_to_csc))
    JDSparseArray(pattern,CuArray(c.nzval),CUDA.zeros(Float64,n),CUDA.ones(Float64,m),0.0)
end
linear_device(::JDSparseArray)=:gpu
linear_sync(::JDSparseArray)=CUDA.synchronize()
linear_default_solver(::JDSparseArray)=:cudss
linear_arrays(c::JDSparseArray,w::CUDA.CuVector{Float64})=(c,w)
linear_arrays(c::JDSparseArray,w)=throw(ArgumentError("JD budgets must be resident Float64 CUDA vectors"))
linear_supports(::JDSparseArray)=nothing
linear_validate_solver(c::JDSparseArray,method)=linear_validate_solver(c.values,method)
linear_validate_solver(c::LinearCUSPARSE.CuSparseMatrixCSR,method)=
    method in (:cudss,:pcg_ic) ? nothing : throw(ArgumentError("invalid sparse GPU solver"))
function linear_validate_market(c::JDSparseArray,w)
    all(x->isfinite(x)&&x>0,c.values) || error("JD utilities must be positive on their stored support")
    all(x->isfinite(x)&&x>0,w) || error("JD budgets must be positive and finite")
    minimum(diff(c.pattern.colptr))>0 && minimum(diff(c.pattern.rowptr))>0 || error("empty JD buyer/product")
end
jd_launch(kernel,n,args...) = (CUDA.@cuda threads=256 blocks=cld(n,256) kernel(args...,n);nothing)

function jd_row_sum_kernel!(out,rp,ids,cols,values,base_rows,base_cols,base_total,n)
    j=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if j<=n
        value=base_rows[j]*base_total
        for q in rp[j]:rp[j+1]-1
            value+=values[ids[q]]-base_rows[j]*base_cols[cols[q]]
        end
        out[j]=value
    end
    return
end
function jd_demand!(out,x::JDSparseArray)
    p=x.pattern
    jd_launch(jd_row_sum_kernel!,p.n,out,p.rowptr,p.csr_to_csc,p.colval,
        x.values,x.base_rows,x.base_cols,sum(x.base_cols))
    out
end
linear_demand(x::JDSparseArray)=jd_demand!(CUDA.zeros(Float64,size(x,1)),x)

function linear_response_workspace(c::JDSparseArray,w)
    (;ratios=similar(c.values),gamma=similar(c.values),counts=CUDA.zeros(Int32,size(c,2)),
      diagonal_edges=similar(c.values),sensitivity_edges=similar(c.values))
end
function jd_response_kernel!(x,s,lam,delta,ratios,counts,cp,rows,c,w,p,sigma,ngoods,rtol,maxiter,m)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=m
        first,last=cp[i],cp[i+1]-1
        best=0.0
        for k in first:last;best=max(best,c[k]/p[rows[k]]);end
        for k in first:last;ratios[k]=(c[k]/p[rows[k]])/best;end
        tau=sigma/w[i];lo=tau/(1+ngoods*tau);hi=min(1.0,ngoods*lo)
        z=isfinite(delta[i])&&delta[i]>0 ? clamp(delta[i],lo,hi) : (lo+hi)/2
        converged=false;used=0
        for it in 1:maxiter
            used=it;mass=0.0;derivative=0.0
            for k in first:last
                r=ratios[k];den=(1-r)+z*r
                mass+=tau*(1-z)*r/den
                derivative-=tau*r/(den*den)
            end
            residual=mass-1
            if abs(residual)<=rtol;converged=true;break;end
            if residual>0;lo=z;else;hi=z;end
            proposal=z-residual/derivative
            z=lo<proposal<hi ? proposal : (residual<0 ? lo : (lo+hi)/2)
        end
        counts[i]=converged ? used : -used
        delta[i]=z;lam[i]=(1-z)/best
        for k in first:last
            s[k]=p[rows[k]]*((1-ratios[k])+z*ratios[k])
            x[k]=sigma/s[k]
        end
    end
    return
end
function linear_perturbed_response!(x::JDSparseArray,s,lam,delta,c,w,p,sigma;
    rtol=1e-12,maxiter=100,supports=nothing,workspace=nothing)
    ws=workspace===nothing ? linear_response_workspace(c,w) : workspace
    pat=c.pattern
    jd_launch(jd_response_kernel!,pat.m,x.values,s.values,lam,delta,ws.ratios,ws.counts,
        pat.colptr,pat.rowval,c.values,w,p,sigma,pat.n,rtol,maxiter)
    x.base_rows .= sigma ./ p;fill!(x.base_cols,1.0);x.zero_spending=sigma
    copyto!(s.base_rows,p);fill!(s.base_cols,1.0);s.zero_spending=0.0
    minimum(ws.counts)>0 || error("sparse perturbed response failed at sigma=$sigma")
    sum(Int64,ws.counts)
end

function linear_metrics_workspace(c::JDSparseArray)
    n,m=size(c)
    (;demand=CUDA.zeros(Float64,n),spend=CUDA.zeros(Float64,m),utility=CUDA.zeros(Float64,m),
      best=CUDA.zeros(Float64,m),budget=CUDA.zeros(Float64,m),regret=CUDA.zeros(Float64,m),
      y=CUDA.zeros(Float64,m),gap=CUDA.zeros(Float64,m),
      dual_violation=CUDA.zeros(Float64,n),dual_signed=CUDA.zeros(Float64,n),slack=CUDA.zeros(Float64,n))
end
function jd_buyer_metrics_kernel!(spend,utility,best,budget,regret,y,cp,rows,c,x,w,p,br,bc,base_spending,m)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=m
        spent=bc[i]*base_spending;u=0.0;bang=0.0
        for k in cp[i]:cp[i+1]-1
            j=rows[k]
            spent+=p[j]*(x[k]-br[j]*bc[i]);u+=c[k]*x[k];bang=max(bang,c[k]/p[j])
        end
        spend[i]=spent;utility[i]=u;best[i]=bang
        budget[i]=abs(spent/w[i]-1);regret[i]=max(0.0,1-u/(w[i]*bang));y[i]=w[i]/u
    end
    return
end
function jd_gap_kernel!(gap,cp,rows,c,x,p,y,base_cols,zero_spending,ngoods,m)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=m
        value=cp[i+1]-cp[i]<ngoods ? zero_spending*base_cols[i] : 0.0
        for k in cp[i]:cp[i+1]-1
            value=max(value,x[k]*max(0.0,p[rows[k]]-c[k]*y[i]))
        end
        gap[i]=value
    end
    return
end
function jd_dual_kernel!(negative,signed,slack,rp,ids,cols,c,p,y,nbuyers,n)
    j=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if j<=n
        largest=0.0;smallest=rp[j+1]-rp[j]<nbuyers ? 0.0 : Inf
        for q in rp[j]:rp[j+1]-1
            cy=c[ids[q]]*y[cols[q]];largest=max(largest,cy);smallest=min(smallest,cy)
        end
        signed[j]=p[j]-largest;negative[j]=max(0.0,largest-p[j]);slack[j]=max(0.0,p[j]-smallest)
    end
    return
end
function linear_equilibrium_metrics(c::JDSparseArray,w,p,x;demand=nothing,workspace=nothing)
    ws=workspace===nothing ? linear_metrics_workspace(c) : workspace
    demand=demand===nothing ? ws.demand : demand
    pat=c.pattern
    jd_demand!(demand,x)
    jd_launch(jd_buyer_metrics_kernel!,pat.m,ws.spend,ws.utility,ws.best,ws.budget,ws.regret,ws.y,
        pat.colptr,pat.rowval,c.values,x.values,w,p,x.base_rows,x.base_cols,dot(p,x.base_rows))
    jd_launch(jd_gap_kernel!,pat.m,ws.gap,pat.colptr,pat.rowval,c.values,x.values,p,ws.y,
        x.base_cols,x.zero_spending,pat.n)
    jd_launch(jd_dual_kernel!,pat.n,ws.dual_violation,ws.dual_signed,ws.slack,
        pat.rowptr,pat.csr_to_csc,pat.colval,c.values,p,ws.y,pat.m)
    mc=maximum(z->abs(z-1),demand);budget=maximum(ws.budget);regret=maximum(ws.regret)
    total_spending=sum(ws.spend);total_budget=sum(w)
    paper_primal=mc/(1+max(maximum(abs,demand),1.0))
    # Literal relative KKT criterion in the paper, with t=c'x and y=w/t.
    paper_dual=maximum(ws.dual_violation)/(1+max(maximum(ws.y),maximum(ws.dual_signed)))
    xmax=max(maximum(x.values),maximum(x.base_rows)*maximum(x.base_cols))
    paper_gap=maximum(ws.gap)/(1+max(xmax,maximum(ws.slack)))
    paper_error=max(paper_primal,paper_dual,paper_gap)
    valid=all(isfinite,p)&&all(>(0),p)&&all(isfinite,x.values)&&all(>=(0),x.values)&&
          all(isfinite,ws.y)&&all(>(0),ws.utility)
    if !valid
        mc=budget=regret=paper_primal=paper_dual=paper_gap=paper_error=Inf
    end
    (;mc,budget,regret,error=max(mc,budget,regret),total_spending,total_budget,
      total_budget_error=abs(total_spending/total_budget-1),paper_primal,paper_dual,paper_gap,paper_error)
end

function jd_uniform_kernel!(values,cp,w,total,m)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=m
        for k in cp[i]:cp[i+1]-1;values[k]=w[i]/total;end
    end
    return
end
function linear_pr_initial_allocation(c::JDSparseArray,w,p)
    x=similar(c);total=sum(w)
    fill!(x.base_rows,1.0);x.base_cols .= w./total;x.zero_spending=total/size(c,1)
    jd_launch(jd_uniform_kernel!,size(c,2),x.values,c.pattern.colptr,w,total)
    x
end
linear_pr_workspace(c::JDSparseArray,p)=nothing
function jd_pr_bids_kernel!(x,cp,c,w,m)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=m
        u=0.0
        for k in cp[i]:cp[i+1]-1;u+=c[k]*x[k];end
        for k in cp[i]:cp[i+1]-1;x[k]=w[i]*c[k]*x[k]/u;end
    end
    return
end
function jd_pr_allocate_kernel!(x,rows,p,count)
    k=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if k<=count;x[k]/=p[rows[k]];end
    return
end
function pr_step!(x::JDSparseArray,p,c,w,ws)
    jd_launch(jd_pr_bids_kernel!,size(c,2),x.values,c.pattern.colptr,c.values,w)
    fill!(x.base_rows,0.0);fill!(x.base_cols,1.0);x.zero_spending=0.0
    jd_demand!(p,x)
    jd_launch(jd_pr_allocate_kernel!,length(x.values),x.values,c.pattern.rowval,p)
    nothing
end

mutable struct JDHessian{A,I} <: AbstractMatrix{Float64}
    matrix::A
    diagonal_indices::I
end
Base.size(H::JDHessian)=size(H.matrix)
Base.copy(H::JDHessian)=JDHessian(copy(H.matrix),H.diagonal_indices)
linear_sync(::JDHessian)=CUDA.synchronize()
struct JDRankFactors{A}
    matrix::A
    transposed::A
end
struct JDSolverWorkspace{S,H,V}
    solver::S
    H::H
    V::V
end
function jd_diagonal_kernel!(indices,rowptr,colval,n)
    j=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if j<=n
        found=Int32(0)
        for q in rowptr[j]:rowptr[j+1]-1
            if colval[q]==j;found=q;break;end
        end
        indices[j]=found
    end
    return
end
function jd_lower_counts_kernel!(counts,rowptr,colval,n)
    j=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if j<=n
        count=Int32(0)
        for q in rowptr[j]:rowptr[j+1]-1
            if colval[q]<=j;count+=1;end
        end
        counts[j]=count
    end
    return
end
function jd_lower_fill_kernel!(columns,indices,pointers,rowptr,colval,n)
    j=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if j<=n
        target=pointers[j]
        for q in rowptr[j]:rowptr[j+1]-1
            if colval[q]<=j
                columns[target]=colval[q];indices[target]=q;target+=1
            end
        end
    end
    return
end
function linear_solver_workspace(c::JDSparseArray,w,method)
    linear_validate_solver(c,method)
    p=c.pattern;n,m=size(c)
    values=CUDA.ones(Float64,length(c.values))
    V=JDRankFactors(LinearCUSPARSE.CuSparseMatrixCSR(p.rowptr,p.colval,copy(values),(n,m)),
        LinearCUSPARSE.CuSparseMatrixCSR(p.colptr,p.rowval,values,(m,n)))
    matrix=LinearCUSPARSE.gemm('N','N',1.0,V.matrix,V.transposed,'O')
    diagonal_indices=CUDA.zeros(Int32,n)
    jd_launch(jd_diagonal_kernel!,n,diagonal_indices,matrix.rowPtr,matrix.colVal)
    minimum(diagonal_indices)>0 || error("JD Hessian has a missing diagonal")
    H=JDHessian(matrix,diagonal_indices)
    counts=CUDA.zeros(Int32,n)
    jd_launch(jd_lower_counts_kernel!,n,counts,matrix.rowPtr,matrix.colVal)
    pointers=CUDA.ones(Int32,n+1)
    view(pointers,2:n+1) .= Int32(1) .+ cumsum(counts)
    count=Int(sum(Int64,counts))
    columns=CUDA.zeros(Int32,count);indices=similar(columns)
    jd_launch(jd_lower_fill_kernel!,n,columns,indices,pointers,matrix.rowPtr,matrix.colVal)
    A=LinearCUSPARSE.CuSparseMatrixCSR(pointers,columns,CUDA.zeros(Float64,count),(n,n))
    vec()=CUDA.zeros(Float64,n)
    direction=vec();cg_type=isdefined(Krylov,:CgWorkspace) ? Krylov.CgWorkspace : Krylov.CgSolver
    cg=method==:pcg_ic ? cg_type(n,n,typeof(direction)) : nothing
    direct=method==:cudss ? CUDSS.CudssSolver(A,"SPD",'L') : nothing
    core=LinearGPUSolver(method,A,indices,direction,vec(),vec(),vec(),vec(),cg,direct,nothing,nothing,1.0,false)
    JDSolverWorkspace(core,H,V)
end
function linear_ipm_buffers(c::JDSparseArray,w,ws::JDSolverWorkspace)
    (;x=similar(c),s=similar(c),H=ws.H,V=ws.V,diagonal=CUDA.zeros(Float64,size(c,1)),
      sensitivity=CUDA.zeros(Float64,size(c,1)))
end
function jd_hessian_edges_kernel!(vv,vvt,de,se,gamma,cp,rows,map,c,x,w,p,sigma,m)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=m
        first,last=cp[i],cp[i+1]-1;u=0.0
        for k in first:last;u+=c[k]*x[k];end
        largest_k=first;largest=0.0
        for k in first:last
            g=c[k]*x[k]/u;gamma[k]=g
            if g>largest;largest=g;largest_k=k;end
        end
        rest=0.0;rest_mass=0.0
        for k in first:last
            if k!=largest_k;rest+=gamma[k]^2;rest_mass+=gamma[k];end
        end
        tau=sigma/w[i];den=tau+largest^2+rest
        for k in first:last
            j=rows[k];g=gamma[k];spend=p[j]*x[k]
            others=k==largest_k ? rest : largest^2+max(0.0,rest-g^2)
            de[k]=spend^2*(tau+others)/(sigma*den)-sigma
            v=spend*g/sqrt(sigma*den);vvt[k]=v;vv[map[k]]=v
            numerator=k==largest_k ? tau+rest-largest*rest_mass : den-g
            se[k]=x[k]*numerator/den-sigma/p[j]
        end
    end
    return
end
function jd_hessian_rows_kernel!(diagonal,sensitivity,rp,ids,de,se,p,sigma,nbuyers,n)
    j=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if j<=n
        d=nbuyers*sigma;s=d/p[j]
        for q in rp[j]:rp[j+1]-1
            k=ids[q];d+=de[k];s+=se[k]
        end
        diagonal[j]=d;sensitivity[j]=s
    end
    return
end
function linear_perturbed_hessian!(H::JDHessian,V,diagonal,c,w,p,x,sigma;
    sensitivity=nothing,supports=nothing,workspace=nothing)
    ws=workspace===nothing ? linear_response_workspace(c,w) : workspace
    sensitivity===nothing && (sensitivity=CUDA.zeros(Float64,size(c,1)))
    pat=c.pattern
    jd_launch(jd_hessian_edges_kernel!,pat.m,V.matrix.nzVal,V.transposed.nzVal,
        ws.diagonal_edges,ws.sensitivity_edges,ws.gamma,pat.colptr,pat.rowval,pat.csc_to_csr,
        c.values,x.values,w,p,sigma)
    jd_launch(jd_hessian_rows_kernel!,pat.n,diagonal,sensitivity,pat.rowptr,pat.csr_to_csc,
        ws.diagonal_edges,ws.sensitivity_edges,p,sigma,pat.m)
    LinearCUSPARSE.gemm!('N','N',-1.0,V.matrix,V.transposed,0.0,H.matrix,'O')
    linear_set_diagonal!(H,diagonal)
    H
end
function linear_set_diagonal!(H::JDHessian,diagonal)
    view(H.matrix.nzVal,H.diagonal_indices) .= diagonal
    H
end
function linear_shift_diagonal!(H::JDHessian,shift)
    view(H.matrix.nzVal,H.diagonal_indices) .+= shift
    H
end
linear_gpu_update_values!(ws,B::LinearCUSPARSE.CuSparseMatrixCSR) =
    (ws.matrix.nzVal .= view(B.nzVal,ws.indices))
function linear_perturbed_solve(H::JDHessian,b;workspace,kwargs...)
    linear_gpu_solve(H.matrix,b;workspace=workspace.solver,kwargs...)
end

# Generic sparse storage for linear PR/IPM on CPU and GPU. Stored values cover
# positive utility entries; a rank-one baseline represents all other allocations.
# In particular x[j,i]=sigma/p[j] at zero utility is retained exactly.
struct LinearSparsePattern{I}
    n::Int
    m::Int
    colptr::I
    rowval::I
    rowptr::I
    colval::I
    csc_to_csr::I
    csr_to_csc::I
end
mutable struct LinearSparseArray{P,V} <: AbstractMatrix{Float64}
    pattern::P
    values::V
    base_rows::V
    base_cols::V
    zero_spending::Float64 # p[j]*base_rows[j], constant for our implicit states
end
Base.size(a::LinearSparseArray) = (a.pattern.n,a.pattern.m)
function Base.similar(a::LinearSparseArray)
    LinearSparseArray(a.pattern,linear_zeros(a.values,length(a.values)),
        linear_zeros(a.values,size(a,1)),fill!(similar(a.values,size(a,2)),1.0),0.0)
end
function Base.copy(a::LinearSparseArray)
    b=similar(a);linear_copy!(b,a);b
end
function linear_copy!(a::LinearSparseArray,b::LinearSparseArray)
    a.pattern===b.pattern || throw(ArgumentError("incompatible sparse allocation patterns"))
    copyto!(a.values,b.values);copyto!(a.base_rows,b.base_rows);copyto!(a.base_cols,b.base_cols)
    a.zero_spending=b.zero_spending
    a
end
function linear_sparse_array(c::SparseMatrixCSC{Float64})
    c=dropzeros!(copy(c))
    n,m=size(c);count=nnz(c)
    max(n,m,count+1)<typemax(Int32) || error("sparse indices exceed Int32 capacity")
    rowcounts=zeros(Int32,n)
    for j in c.rowval;rowcounts[j]+=1;end
    rowptr=Int32.(vcat(1,1 .+ cumsum(rowcounts)))
    next=copy(rowptr[1:n]);colval=Vector{Int32}(undef,count)
    csc_to_csr=similar(colval);csr_to_csc=similar(colval)
    for i in 1:m, k in c.colptr[i]:c.colptr[i+1]-1
        j=c.rowval[k];q=next[j];next[j]+=1
        colval[q]=i;csc_to_csr[k]=q;csr_to_csc[q]=k
    end
    pattern=LinearSparsePattern(n,m,copy(Int32.(c.colptr)),copy(Int32.(c.rowval)),
        copy(rowptr),copy(colval),copy(csc_to_csr),copy(csr_to_csc))
    LinearSparseArray(pattern,copy(c.nzval),zeros(Float64,n),ones(Float64,m),0.0)
end
const LinearSparseCPUArray = LinearSparseArray{P,Vector{Float64}} where P
linear_device(c::LinearSparseArray)=linear_device(c.values)
linear_sync(c::LinearSparseArray)=linear_sync(c.values)
linear_default_solver(c::LinearSparseArray)=linear_default_solver(c.values)
function linear_arrays(c::LinearSparseArray,w::AbstractVector{Float64})
    linear_device(c)==linear_device(w) || throw(ArgumentError("mixed CPU/GPU market arrays"))
    c,w
end
linear_supports(::LinearSparseArray)=nothing
linear_validate_solver(c::LinearSparseArray,method)=linear_validate_solver(c.values,method)
linear_zeros(c::LinearSparseArray,dims...)=linear_zeros(c.values,dims...)

"""Prepare raw utilities/budgets once; all algorithms receive the same array types."""
function linear_prepare_market(c,w;device=:cpu,storage=:sparse)
    device in (:cpu,:gpu) || throw(ArgumentError("device must be cpu or gpu"))
    storage in (:dense,:sparse) || throw(ArgumentError("storage must be dense or sparse"))
    matrix=storage==:sparse ? linear_sparse_array(SparseMatrixCSC{Float64,Int}(c)) : Matrix{Float64}(c)
    budgets=Vector{Float64}(w)
    matrix=linear_to_device(matrix,Val(device));budgets=linear_to_device(budgets,Val(device))
    linear_validate_market(matrix,budgets)
    (;c=matrix,w=budgets)
end
linear_to_device(a,::Val{:cpu})=a
linear_sparse_launch(kernel,n,args...) = linear_sparse_launch(first(args),kernel,n,args...)
function linear_sparse_launch(::Vector,kernel::F,n,args...) where F
    @inbounds for index in 1:n
        kernel(index,args...,n)
    end
    nothing
end
function linear_validate_market(c::LinearSparseArray,w)
    all(x->isfinite(x)&&x>0,c.values) || error("sparse utilities must be positive on their stored support")
    all(x->isfinite(x)&&x>0,w) || error("sparse budgets must be positive and finite")
    minimum(diff(c.pattern.colptr))>0 && minimum(diff(c.pattern.rowptr))>0 || error("empty sparse buyer/product")
end

function linear_sparse_row_sum_kernel!(j,out,rp,ids,cols,values,base_rows,base_cols,base_total,n)
    @inbounds if j<=n
        value=base_rows[j]*base_total
        for q in rp[j]:rp[j+1]-1
            value+=values[ids[q]]-base_rows[j]*base_cols[cols[q]]
        end
        out[j]=value
    end
    return
end
function linear_sparse_demand!(out,x::LinearSparseArray)
    p=x.pattern
    linear_sparse_launch(linear_sparse_row_sum_kernel!,p.n,out,p.rowptr,p.csr_to_csc,p.colval,
        x.values,x.base_rows,x.base_cols,sum(x.base_cols))
    out
end
linear_demand(x::LinearSparseArray)=linear_sparse_demand!(linear_zeros(x.values,size(x,1)),x)

function linear_response_workspace(c::LinearSparseArray,w)
    (;ratios=similar(c.values),gamma=similar(c.values),counts=fill!(similar(c.values,Int32,size(c,2)),0),
      diagonal_edges=similar(c.values),sensitivity_edges=similar(c.values))
end
function linear_sparse_response_kernel!(i,x,s,lam,delta,ratios,counts,cp,rows,c,w,p,sigma,ngoods,rtol,maxiter,m)
    @inbounds if i<=m
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
function linear_perturbed_response!(x::LinearSparseArray,s,lam,delta,c,w,p,sigma;
    rtol=1e-12,maxiter=100,supports=nothing,workspace=nothing)
    ws=workspace===nothing ? linear_response_workspace(c,w) : workspace
    pat=c.pattern
    linear_sparse_launch(linear_sparse_response_kernel!,pat.m,x.values,s.values,lam,delta,ws.ratios,ws.counts,
        pat.colptr,pat.rowval,c.values,w,p,sigma,pat.n,rtol,maxiter)
    x.base_rows .= sigma ./ p;fill!(x.base_cols,1.0);x.zero_spending=sigma
    copyto!(s.base_rows,p);fill!(s.base_cols,1.0);s.zero_spending=0.0
    minimum(ws.counts)>0 || error("sparse perturbed response failed at sigma=$sigma")
    sum(Int64,ws.counts)
end

function linear_metrics_workspace(c::LinearSparseArray;kkt=false)
    n,m=size(c)
    extras=kkt ? (;y=linear_zeros(c.values,m),gap=linear_zeros(c.values,m),
        dual_violation=linear_zeros(c.values,n),dual_signed=linear_zeros(c.values,n),slack=linear_zeros(c.values,n)) : nothing
    (;demand=linear_zeros(c.values,n),spend=linear_zeros(c.values,m),utility=linear_zeros(c.values,m),
      best=linear_zeros(c.values,m),budget=linear_zeros(c.values,m),regret=linear_zeros(c.values,m),kkt=extras)
end
function linear_sparse_buyer_metrics_kernel!(i,spend,utility,best,budget,regret,y,cp,rows,c,x,w,p,br,bc,base_spending,m)
    @inbounds if i<=m
        spent=bc[i]*base_spending;u=0.0;bang=0.0
        for k in cp[i]:cp[i+1]-1
            j=rows[k]
            spent+=p[j]*(x[k]-br[j]*bc[i]);u+=c[k]*x[k];bang=max(bang,c[k]/p[j])
        end
        spend[i]=spent;utility[i]=u;best[i]=bang
        budget[i]=abs(spent/w[i]-1);regret[i]=max(0.0,1-u/(w[i]*bang))
        y===nothing || (y[i]=w[i]/u)
    end
    return
end
function linear_sparse_gap_kernel!(i,gap,cp,rows,c,x,p,y,base_cols,zero_spending,ngoods,m)
    @inbounds if i<=m
        value=cp[i+1]-cp[i]<ngoods ? zero_spending*base_cols[i] : 0.0
        for k in cp[i]:cp[i+1]-1
            value=max(value,x[k]*max(0.0,p[rows[k]]-c[k]*y[i]))
        end
        gap[i]=value
    end
    return
end
function linear_sparse_dual_kernel!(j,negative,signed,slack,rp,ids,cols,c,p,y,nbuyers,n)
    @inbounds if j<=n
        largest=0.0;smallest=rp[j+1]-rp[j]<nbuyers ? 0.0 : Inf
        for q in rp[j]:rp[j+1]-1
            cy=c[ids[q]]*y[cols[q]];largest=max(largest,cy);smallest=min(smallest,cy)
        end
        signed[j]=p[j]-largest;negative[j]=max(0.0,largest-p[j]);slack[j]=max(0.0,p[j]-smallest)
    end
    return
end
function linear_equilibrium_metrics(c::LinearSparseArray,w,p,x;demand=nothing,workspace=nothing)
    ws=workspace===nothing ? linear_metrics_workspace(c) : workspace
    demand=demand===nothing ? ws.demand : demand
    pat=c.pattern
    linear_sparse_demand!(demand,x)
    y=ws.kkt===nothing ? nothing : ws.kkt.y
    linear_sparse_launch(linear_sparse_buyer_metrics_kernel!,pat.m,ws.spend,ws.utility,ws.best,ws.budget,ws.regret,y,
        pat.colptr,pat.rowval,c.values,x.values,w,p,x.base_rows,x.base_cols,dot(p,x.base_rows))
    mc=maximum(z->abs(z-1),demand);budget=maximum(ws.budget);regret=maximum(ws.regret)
    total_spending=sum(ws.spend);total_budget=sum(w)
    valid=all(isfinite,p)&&all(>(0),p)&&all(isfinite,x.values)&&all(>=(0),x.values)&&
          all(isfinite,ws.utility)&&all(>(0),ws.utility)
    valid || (mc=budget=regret=Inf)
    shared=(;mc,budget,regret,error=max(mc,budget,regret),total_spending,total_budget,
        total_budget_error=abs(total_spending/total_budget-1))
    ws.kkt===nothing && return shared
    extra=ws.kkt
    linear_sparse_launch(linear_sparse_gap_kernel!,pat.m,extra.gap,pat.colptr,pat.rowval,c.values,x.values,p,extra.y,
        x.base_cols,x.zero_spending,pat.n)
    linear_sparse_launch(linear_sparse_dual_kernel!,pat.n,extra.dual_violation,extra.dual_signed,extra.slack,
        pat.rowptr,pat.csr_to_csc,pat.colval,c.values,p,extra.y,pat.m)
    paper_primal=mc/(1+max(maximum(abs,demand),1.0))
    paper_dual=maximum(extra.dual_violation)/(1+max(maximum(extra.y),maximum(extra.dual_signed)))
    xmax=max(maximum(x.values),maximum(x.base_rows)*maximum(x.base_cols))
    paper_gap=maximum(extra.gap)/(1+max(xmax,maximum(extra.slack)))
    paper_error=max(paper_primal,paper_dual,paper_gap)
    valid || (paper_primal=paper_dual=paper_gap=paper_error=Inf)
    merge(shared,(;paper_primal,paper_dual,paper_gap,paper_error))
end

function linear_sparse_uniform_kernel!(i,values,cp,w,total,m)
    @inbounds if i<=m
        for k in cp[i]:cp[i+1]-1;values[k]=w[i]/total;end
    end
    return
end
function linear_pr_initial_allocation(c::LinearSparseArray,w,p)
    x=similar(c);total=sum(w)
    fill!(x.base_rows,1.0);x.base_cols .= w./total;x.zero_spending=total/size(c,1)
    linear_sparse_launch(linear_sparse_uniform_kernel!,size(c,2),x.values,c.pattern.colptr,w,total)
    x
end
linear_pr_workspace(c::LinearSparseArray,p)=nothing
function linear_sparse_pr_bids_kernel!(i,x,cp,c,w,m)
    @inbounds if i<=m
        u=0.0
        for k in cp[i]:cp[i+1]-1;u+=c[k]*x[k];end
        for k in cp[i]:cp[i+1]-1;x[k]=w[i]*c[k]*x[k]/u;end
    end
    return
end
function linear_sparse_pr_allocate_kernel!(k,x,rows,p,count)
    @inbounds if k<=count;x[k]/=p[rows[k]];end
    return
end
function pr_step!(x::LinearSparseArray,p,c,w,ws)
    linear_sparse_launch(linear_sparse_pr_bids_kernel!,size(c,2),x.values,c.pattern.colptr,c.values,w)
    fill!(x.base_rows,0.0);fill!(x.base_cols,1.0);x.zero_spending=0.0
    linear_sparse_demand!(p,x)
    linear_sparse_launch(linear_sparse_pr_allocate_kernel!,length(x.values),x.values,c.pattern.rowval,p)
    nothing
end

mutable struct LinearSparseHessian{A,I} <: AbstractMatrix{Float64}
    matrix::A
    diagonal_indices::I
end
Base.size(H::LinearSparseHessian)=size(H.matrix)
Base.copy(H::LinearSparseHessian)=LinearSparseHessian(copy(H.matrix),H.diagonal_indices)
linear_sync(H::LinearSparseHessian)=linear_sync(H.diagonal_indices)
struct LinearSparseRankFactors{A}
    matrix::A
    transposed::A
end
struct LinearSparseSolverWorkspace{S,H,V}
    solver::S
    H::H
    V::V
end
Base.copy(V::LinearSparseRankFactors)=LinearSparseRankFactors(copy(V.matrix),copy(V.transposed))
function linear_sparse_diagonal_kernel!(j,indices,rowptr,colval,n)
    @inbounds if j<=n
        found=Int32(0)
        for q in rowptr[j]:rowptr[j+1]-1
            if colval[q]==j;found=q;break;end
        end
        indices[j]=found
    end
    return
end
function linear_sparse_lower_counts_kernel!(j,counts,rowptr,colval,n)
    @inbounds if j<=n
        count=Int32(0)
        for q in rowptr[j]:rowptr[j+1]-1
            if colval[q]<=j;count+=1;end
        end
        counts[j]=count
    end
    return
end
function linear_sparse_lower_fill_kernel!(j,columns,indices,pointers,rowptr,colval,n)
    @inbounds if j<=n
        target=pointers[j]
        for q in rowptr[j]:rowptr[j+1]-1
            if colval[q]<=j
                columns[target]=colval[q];indices[target]=q;target+=1
            end
        end
    end
    return
end
function linear_ipm_buffers(c::LinearSparseArray,w,ws::LinearSparseSolverWorkspace)
    (;x=similar(c),s=similar(c),H=ws.H,V=ws.V,diagonal=linear_zeros(c.values,size(c,1)),
      sensitivity=linear_zeros(c.values,size(c,1)))
end
function linear_sparse_hessian_edges_kernel!(i,vv,vvt,de,se,gamma,cp,rows,map,c,x,w,p,sigma,m)
    @inbounds if i<=m
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
function linear_sparse_hessian_rows_kernel!(j,diagonal,sensitivity,rp,ids,de,se,p,sigma,nbuyers,n)
    @inbounds if j<=n
        d=nbuyers*sigma;s=d/p[j]
        for q in rp[j]:rp[j+1]-1
            k=ids[q];d+=de[k];s+=se[k]
        end
        diagonal[j]=d;sensitivity[j]=s
    end
    return
end
function linear_perturbed_hessian!(H::LinearSparseHessian,V,diagonal,c,w,p,x,sigma;
    sensitivity=nothing,supports=nothing,workspace=nothing)
    ws=workspace===nothing ? linear_response_workspace(c,w) : workspace
    sensitivity===nothing && (sensitivity=linear_zeros(c.values,size(c,1)))
    pat=c.pattern
    vv,vvt=linear_sparse_rank_values(V)
    linear_sparse_launch(linear_sparse_hessian_edges_kernel!,pat.m,vv,vvt,
        ws.diagonal_edges,ws.sensitivity_edges,ws.gamma,pat.colptr,pat.rowval,pat.csc_to_csr,
        c.values,x.values,w,p,sigma)
    linear_sparse_launch(linear_sparse_hessian_rows_kernel!,pat.n,diagonal,sensitivity,pat.rowptr,pat.csr_to_csc,
        ws.diagonal_edges,ws.sensitivity_edges,p,sigma,pat.m)
    linear_sparse_product!(H,V)
    linear_set_diagonal!(H,diagonal)
    H
end
function linear_set_diagonal!(H::LinearSparseHessian,diagonal)
    view(nonzeros(H.matrix),H.diagonal_indices) .= diagonal
    H
end
function linear_shift_diagonal!(H::LinearSparseHessian,shift)
    view(nonzeros(H.matrix),H.diagonal_indices) .+= shift
    H
end

# CPU sparse assembly and factorization. No goods × buyers dense workspace.
mutable struct LinearSparseCPUFactorization
    factor::Any
end
function linear_solver_workspace(c::LinearSparseCPUArray,w,method)
    method in (:cholesky,:pcg_ic) || throw(ArgumentError("sparse CPU solver must be :cholesky or :pcg_ic"))
    p=c.pattern;n,m=size(c)
    V=SparseMatrixCSC(n,m,Int.(p.colptr),Int.(p.rowval),ones(length(c.values)))
    factors=LinearSparseRankFactors(V,copy(transpose(V)))
    matrix=V*factors.transposed
    diagonal_indices=[searchsortedfirst(view(matrix.rowval,nzrange(matrix,j)),j)+matrix.colptr[j]-1 for j in 1:n]
    H=LinearSparseHessian(matrix,diagonal_indices)
    LinearSparseSolverWorkspace(LinearSparseCPUFactorization(nothing),H,factors)
end
linear_sparse_rank_values(V::LinearSparseRankFactors{<:SparseMatrixCSC})=(nonzeros(V.transposed),nonzeros(V.matrix))
function linear_sparse_product!(H::LinearSparseHessian{<:SparseMatrixCSC},V)
    product=V.matrix*V.transposed
    product.colptr==H.matrix.colptr && product.rowval==H.matrix.rowval || error("sparse Hessian pattern changed")
    H.matrix.nzval .= .-product.nzval
    H
end
function linear_cholesky(B,ws::LinearSparseCPUFactorization)
    if ws.factor===nothing
        ws.factor=cholesky(B;check=false)
    else
        cholesky!(ws.factor,B;check=false)
    end
    ws.factor
end
function linear_perturbed_solve(H::LinearSparseHessian{<:SparseMatrixCSC},b;workspace,kwargs...)
    linear_perturbed_solve(H.matrix,b;workspace=workspace.solver,kwargs...)
end

# Right-looking IC(0) updates only entries in the original lower CSC pattern.
# Sorted-column intersections avoid both dense storage and dense setup loops.
function perturbed_ic(B::Symmetric{Float64,<:SparseMatrixCSC};drop=0.0)
    drop>=0 || throw(ArgumentError("drop must be nonnegative"))
    L=tril(sparse(B));n=size(L,1);d=diag(L)
    all(>(0),d) || error("positive diagonal required")
    if drop>0
        for j in 1:n, k in nzrange(L,j)
            i=L.rowval[k]
            i!=j && abs(L.nzval[k])<drop*sqrt(d[i]*d[j]) && (L.nzval[k]=0.0)
        end
        dropzeros!(L)
    end
    cp,rows,values=L.colptr,L.rowval,L.nzval
    @inbounds for j in 1:n
        first,last=cp[j],cp[j+1]-1
        rows[first]==j || error("missing IC diagonal")
        pivot=values[first]
        pivot>0 || error("incomplete Cholesky failed at pivot $j; no shift applied")
        values[first]=sqrt(pivot)
        for k in first+1:last;values[k]/=values[first];end
        for k in first+1:last
            i=rows[k];lij=values[k];a=k;b=cp[i];bend=cp[i+1]-1
            while a<=last && b<=bend
                ra,rb=rows[a],rows[b]
                if ra==rb
                    values[b]-=values[a]*lij;a+=1;b+=1
                elseif ra<rb
                    a+=1
                else
                    b+=1
                end
            end
        end
    end
    factor=LowerTriangular(L)
    apply(r)=factor' \ (factor \ r)
    (;apply,factor,factor_nnz=nnz(L),factor_order=n,drop)
end

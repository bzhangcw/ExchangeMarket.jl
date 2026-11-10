using Test,Random,SparseArrays,LinearAlgebra
include("linear-perturbed.jl")
include("propres.jl")
include("linear-gpu.jl")
include("linear-jd-gpu.jl")
CUDA.functional() || error("CUDA device required")
BLAS.set_num_threads(1)
function materialize(a::JDSparseArray)
    out=Array(a.base_rows)*Array(a.base_cols)'
    cp,rows,values=Array(a.pattern.colptr),Array(a.pattern.rowval),Array(a.values)
    for i in 1:size(a,2),k in cp[i]:cp[i+1]-1;out[rows[k],i]=values[k];end
    out
end
function paper_reference(c,w,p,x)
    d=vec(sum(x;dims=2));u=vec(sum(c.*x;dims=1));y=w./u
    g=p.-c.*y';positive=max.(g,0.0)
    a=maximum(abs.(d.-1))/(1+max(maximum(abs,d),1.0))
    b=max(0.0,-minimum(g))/(1+max(maximum(y),maximum(vec(minimum(g;dims=2)))))
    cc=maximum(x.*positive)/(1+max(maximum(x),maximum(positive)))
    (;paper_primal=a,paper_dual=b,paper_gap=cc,paper_error=max(a,b,cc))
end
rng=MersenneTwister(12);n,m=19,31
c=Float64.(rand(rng,n,m).<.2).*rand(rng,1:9,n,m)
for j in 1:n;c[j,mod1(j,m)]=1;end
for i in 1:m;c[mod1(i,n),i]=2;end
w=rand(rng,m).+.1;w./=sum(w);p=fill(1/n,n)
gc=jd_gpu_array(sparse(c));gw=CuArray(w);gp=CuArray(p)
@testset "Sparse implicit responses and native Hessian" begin
    ws=linear_response_workspace(gc,gw)
    system=linear_solver_workspace(gc,gw,:cudss)
    buffers=linear_ipm_buffers(gc,gw,system)
    for sigma in (1/(n*(m+1)),1e-4*minimum(w)/n)
        x=zeros(n,m);s=similar(x);lam=zeros(m);delta=zeros(m)
        gl=CUDA.zeros(Float64,m);gd=similar(gl);fill!(gd,0)
        linear_perturbed_response!(x,s,lam,delta,c,w,p,sigma)
        linear_perturbed_response!(buffers.x,buffers.s,gl,gd,gc,gw,gp,sigma;workspace=ws)
        @test materialize(buffers.x) ≈ x rtol=1e-9
        @test materialize(buffers.s) ≈ s rtol=1e-9 atol=1e-13
        @test Array(linear_demand(buffers.x)) ≈ vec(sum(x;dims=2)) rtol=1e-10
        H=zeros(n,n);V=zeros(n,m);d=zeros(n);v=zeros(n)
        linear_perturbed_hessian!(H,V,d,c,w,p,x,sigma;sensitivity=v)
        linear_perturbed_hessian!(buffers.H,buffers.V,buffers.diagonal,gc,gw,gp,buffers.x,sigma;
            sensitivity=buffers.sensitivity,workspace=ws)
        @test Array(CuMatrix(buffers.H.matrix)) ≈ H rtol=1e-9
        @test Array(buffers.sensitivity) ≈ v rtol=1e-8 atol=1e-12
        metrics=linear_equilibrium_metrics(gc,gw,gp,buffers.x)
        ref=linear_equilibrium_metrics(c,w,p,x)
        for name in keys(ref);@test getproperty(metrics,name) ≈ getproperty(ref,name) rtol=1e-8 atol=1e-11;end
        for (name,value) in pairs(paper_reference(c,w,p,x));@test getproperty(metrics,name) ≈ value rtol=1e-8 atol=1e-11;end
    end
end
@testset "Sparse PR retains dense initialization and trajectory" begin
    for k in (0,1,7,30)
        cpu=run_pr(c,w;stop=z->false,maxiter=k)
        gpu=run_pr(gc,gw;stop=z->false,maxiter=k)
        @test materialize(gpu.st.x) ≈ cpu.st.x rtol=1e-9
        @test Array(gpu.st.p) ≈ cpu.st.p rtol=1e-9
        metrics=linear_equilibrium_metrics(gc,gw,gpu.st.p,gpu.st.x)
        for (name,value) in pairs(paper_reference(c,w,cpu.st.p,cpu.st.x));@test getproperty(metrics,name) ≈ value rtol=1e-8 atol=1e-11;end
    end
end
@testset "Same IPM outer algorithm with sparse cuDSS and PCG" begin
    stop=z->z.error<=1e-4*(1+1e-8)
    common=(tol=1e-4,σ_min=1e-4*minimum(w)/n,show_trace=false,maxiter=200,stop=stop)
    cpu=linear_perturbed_ipm((;c,w);common...)
    for solver in (:cudss,:pcg_ic)
        println("Checking sparse ",solver);flush(stdout)
        gpu=linear_perturbed_ipm((c=gc,w=gw);common...,linear_solver=solver)
        @test gpu.status==:converged
        @test Array(gpu.st.p) ≈ cpu.st.p rtol=1e-6
        @test materialize(gpu.st.x) ≈ cpu.st.x rtol=1e-6 atol=1e-9
        @test gpu.traj[end].k==cpu.traj[end].k
        @test all(z.success for z in gpu.linear_stats)
        @test length(gpu.st.x.values)==nnz(sparse(c))
    end
end

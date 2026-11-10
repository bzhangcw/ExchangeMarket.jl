# Run with the standalone GPU project; this test requires an actual CUDA device.
using Test, Random, SparseArrays, LinearAlgebra
include("linear-perturbed.jl")
include("propres.jl")
include("linear-gpu.jl")
CUDA.functional() || error("GPU correctness tests require CUDA")
BLAS.set_num_threads(1)
const n,m=500,1000
rng=MersenneTwister(1)
c=Matrix(sprand(rng,n,m,.02));w=.5 .+rand(rng,m);w./=sum(w)
for i in 1:m; maximum(view(c,:,i))==0 && (c[rand(rng,1:n),i]=1);end
for j in 1:n; maximum(view(c,j,:))==0 && (c[j,rand(rng,1:m)]=1);end
cg,wg=CUDA.CuArray(c),CUDA.CuArray(w)
p=fill(1/n,n);pg=CUDA.CuArray(p)
x=zeros(n,m);s=similar(x);lam=zeros(m);delta=zeros(m)
xg=CUDA.zeros(Float64,n,m);sg=similar(xg);lg=CUDA.zeros(Float64,m);dg=copy(lg)
ws=linear_response_workspace(cg,wg)
@testset "Resident GPU response and Hessian (500 × 1000)" begin
    for sigma in (1/(n*(m+1)),1e-4*minimum(w)/n)
        fill!(delta,0);fill!(dg,0)
        linear_perturbed_response!(x,s,lam,delta,c,w,p,sigma)
        linear_perturbed_response!(xg,sg,lg,dg,cg,wg,pg,sigma;workspace=ws)
        @test Array(xg) ≈ x rtol=1e-9
        @test Array(dg) ≈ delta rtol=1e-9
        H=zeros(n,n);V=zeros(n,m);d=zeros(n);v=zeros(n)
        Hg=CUDA.zeros(Float64,n,n);Vg=similar(xg);dd=similar(pg);vg=similar(pg)
        linear_perturbed_hessian!(H,V,d,c,w,p,x,sigma;sensitivity=v)
        linear_perturbed_hessian!(Hg,Vg,dd,cg,wg,pg,xg,sigma;sensitivity=vg,workspace=ws)
        @test Array(Hg) ≈ H rtol=1e-8
        @test Array(vg) ≈ v rtol=1e-8 atol=1e-12
        a=linear_equilibrium_metrics(c,w,p,x)
        b=linear_equilibrium_metrics(cg,wg,pg,xg)
        for k in keys(a); @test getproperty(a,k) ≈ getproperty(b,k) rtol=1e-8 atol=1e-10;end
    end
    @test linear_arrays(cg,wg)[1] === cg
    @test_throws ArgumentError linear_validate_solver(cg,:cholesky)
end
@testset "GPU PR agrees with CPU at fixed checkpoints" begin
    for k in (1,5,20)
        a=run_pr(c,w;stop=z->false,maxiter=k)
        b=run_pr(cg,wg;stop=z->false,maxiter=k)
        @test b.st.x isa CUDA.CuArray
        @test Array(b.st.x) ≈ a.st.x rtol=1e-10
        @test Array(b.st.p) ≈ a.st.p rtol=1e-10
    end
end
@testset "Shared CPU/GPU IPM and native linear solvers" begin
    stop=z->z.error <= 1e-4*(1+1e-8)
    common=(tol=1e-4,σ_min=1e-4*minimum(w)/n,show_trace=false,maxiter=200,stop=stop)
    cpu=linear_perturbed_ipm((;c,w);common...)
    for solver in (:cudss,:pcg_ic)
        gpu=linear_perturbed_ipm((c=cg,w=wg);common...,linear_solver=solver)
        @test gpu.status == :converged
        @test gpu.st.x isa CUDA.CuArray
        @test gpu.st.p isa CUDA.CuArray
        @test stop(linear_equilibrium_metrics(cg,wg,gpu.st.p,gpu.st.x))
        @test Array(gpu.st.p) ≈ cpu.st.p rtol=1e-6
        @test gpu.traj[end].k == cpu.traj[end].k
        @test all(z.success for z in gpu.linear_stats)
        @test solver == :cudss ? gpu.factorizations>0 : gpu.preconditioner_factorizations>0
    end
end

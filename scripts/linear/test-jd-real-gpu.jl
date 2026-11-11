using Test,Random,SparseArrays,LinearAlgebra
include(joinpath(@__DIR__, "../../dataset/jd-data.jl"))
include("linear-perturbed.jl")
include("propres.jl")
include("linear-gpu.jl")
include("linear-jd-gpu.jl")
BLAS.set_num_threads(1)
f=JDData.load_market(joinpath(@__DIR__,"../../dataset/jd/utilities.bin");users=2000)
c=Matrix(f.c);w=f.w;n,m=size(c);p=fill(sum(w)/n,n)
gc=jd_gpu_array(f.c);gw=CuArray(w);gp=CuArray(p)
ws=linear_response_workspace(gc,gw);system=linear_solver_workspace(gc,gw,:pcg_ic)
buffers=linear_ipm_buffers(gc,gw,system)
sigma=sum(w)/(n*(m+1));lam=CUDA.zeros(Float64,m);delta=similar(lam);fill!(delta,0)
linear_perturbed_response!(buffers.x,buffers.s,lam,delta,gc,gw,gp,sigma;workspace=ws)
x=zeros(n,m);s=similar(x);cl=zeros(m);cd=zeros(m)
linear_perturbed_response!(x,s,cl,cd,c,w,p,sigma)
indices=[f.c.rowval[k]+(i-1)*n for i in 1:m for k in f.c.colptr[i]:f.c.colptr[i+1]-1]
@testset "Actual JD response and Hessian derivative" begin
    @test Array(buffers.x.values)≈x[indices] rtol=1e-9
    @test Array(linear_demand(buffers.x))≈vec(sum(x;dims=2)) rtol=1e-10
    @test Array(delta)≈cd rtol=1e-10
    linear_perturbed_hessian!(buffers.H,buffers.V,buffers.diagonal,gc,gw,gp,buffers.x,sigma;
        sensitivity=buffers.sensitivity,workspace=ws)
    rng=MersenneTwister(15);v=randn(rng,n);v./=maximum(abs,v);gv=CuArray(v)
    h=sigma/maximum(w)/100
    function demand_at(step)
        pp=gp.*(1 .+step.*gv);xx=similar(gc);ss=similar(gc);ll=similar(gw);dd=copy(delta)
        linear_perturbed_response!(xx,ss,ll,dd,gc,gw,pp,sigma;workspace=ws)
        linear_demand(xx)
    end
    numerical=-gp.*(demand_at(h).-demand_at(-h))./(2h)
    exact=buffers.H.matrix*gv
    relative=norm(numerical-exact)/norm(exact)
    println("JD Hessian directional derivative relative error: ",relative)
    @test relative<2e-3
    # The same matrix's columns must be sorted for native IC(0).
    A=system.solver.matrix;rp,ci=Array(A.rowPtr),Array(A.colVal)
    @test all(issorted(ci[rp[j]:rp[j+1]-1]) for j in 1:n)
end

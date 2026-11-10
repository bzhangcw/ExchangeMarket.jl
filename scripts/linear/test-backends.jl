using Test, Random, LinearAlgebra, SparseArrays
include("linear-perturbed.jl")
include("propres.jl")
BLAS.set_num_threads(1)
@testset "Array kernels match CPU reference" begin
    rng=MersenneTwister(9)
    for density in (0.1, 1.0), sigma in (0.1, 1e-9)
        n,m=9,17
        c=Matrix(sprand(rng,n,m,density));c[1,:].+=0.2;c[:,1].+=0.2
        w=0.5 .+rand(rng,m);w./=sum(w);p=fill(1/n,n)
        x=zeros(n,m);s=similar(x);lam=zeros(m);delta=zeros(m)
        xx=similar(x);ss=similar(x);ll=zeros(m);dd=zeros(m)
        ws=linear_array_response_workspace(c,w)
        linear_perturbed_response!(x,s,lam,delta,c,w,p,sigma)
        linear_array_response!(xx,ss,ll,dd,c,w,p,sigma,ws)
        @test xx ≈ x rtol=1e-10
        @test dd ≈ delta rtol=1e-10
        H=zeros(n,n);V=zeros(n,m);di=zeros(n);sens=zeros(n)
        HH=similar(H);VV=similar(V);didi=similar(di);ssens=similar(sens)
        linear_perturbed_hessian!(H,V,di,c,w,p,x,sigma;sensitivity=sens)
        linear_array_hessian!(HH,VV,didi,c,w,p,x,sigma,ws;sensitivity=ssens)
        @test HH ≈ H rtol=1e-10
        @test ssens ≈ sens rtol=1e-10
        metricws=linear_array_metrics_workspace(c)
        a=linear_equilibrium_metrics(c,w,p,x);b=linear_array_metrics(c,w,p,x,metricws)
        for k in keys(a); @test getproperty(a,k) ≈ getproperty(b,k) atol=1e-12;end
        x .= w' .* n;xx.=x;pp=copy(p)
        for _ in 1:5
            pr_step!(x,p,c,w,zeros(n))
            linear_array_pr_step!(xx,pp,c,w,metricws)
        end
        @test xx ≈ x rtol=1e-12
        @test pp ≈ p rtol=1e-12
    end
    @test_throws ArgumentError linear_validate_solver(ones(2,2),:cudss)
    @test linear_default_solver(ones(2,2)) == :cholesky
end
@testset "Tied largest utility shares" begin
    c=ones(4,2);w=[.4,.6];p=fill(.25,4);sigma=1e-8
    x=zeros(4,2);s=similar(x);lam=zeros(2);delta=zeros(2)
    linear_perturbed_response!(x,s,lam,delta,c,w,p,sigma)
    H=zeros(4,4);V=zeros(4,2);d=zeros(4);sens=zeros(4)
    A=similar(H);W=similar(V);dd=similar(d);tt=similar(sens)
    linear_perturbed_hessian!(H,V,d,c,w,p,x,sigma;sensitivity=sens)
    linear_array_hessian!(A,W,dd,c,w,p,x,sigma,linear_array_response_workspace(c,w);sensitivity=tt)
    @test A ≈ H rtol=1e-12
    @test tt ≈ sens rtol=1e-12
end

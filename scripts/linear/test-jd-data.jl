using Test, SparseArrays
include("jd-data.jl")
@testset "JD prefixes retain counts and normalize selected budgets" begin
    mktempdir() do dir
        path=joinpath(dir,"utilities.bin")
        open(path,"w") do io
            write(io,"JDUTIL01",UInt64[1,3,4,6,1,0,0])
            write(io,UInt64[1,3,4,6,7],UInt32[1,2,2,2,3,1])
            write(io,Float64[2,5,1,9,3,4],Float64[.2,.6,.3,.1])
        end
        one=JDData.load_market(path;users=1)
        two=JDData.load_market(path;users=2)
        all=JDData.load_market(path)
        @test one.w==[1.0]
        @test two.w≈[.25,.75]
        @test sum(all.w)≈1
        @test two.n==2 && two.m==2
        @test Matrix(two.c)==[2 0;5 1]
        @test nonzeros(all.c)==[2,5,1,9,3,4]
        @test two.w[1]/two.w[2]≈all.w[1]/all.w[2]
        @test two.utility_normalization=="none"
        @test two.budget_normalization=="sum_one"
        @test_throws ArgumentError JDData.load_market(path;users=0)
        @test_throws ArgumentError JDData.load_market(path;users=5)
        open(path,"a") do io;write(io,0x00);end
        @test_throws ErrorException JDData.load_market(path)
    end
end

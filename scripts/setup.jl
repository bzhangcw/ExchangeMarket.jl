using Pkg
try
    Pkg.rm("ExchangeMarket")
catch
end
Pkg.develop(path=joinpath(@__DIR__, ".."))

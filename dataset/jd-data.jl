# Portable little-endian CSC format written by jd_preprocess.py.
module JDData
using SparseArrays
const PAPER_USERS = [2000,5000,20000,50000,200000,500000,1000000,2557841]
const PAPER_GOODS = [13541,15887,19449,21410,23853,25214,26115,27330]
const PAPER_NNZ = [136877,265220,698579,1278391,2937906,4679648,6189526,8073837]

function load_market(path; users=nothing)
    ENDIAN_BOM == 0x04030201 || error("JDUTIL01 currently requires a little-endian Julia host")
    open(path,"r") do io
        String(read(io,8)) == "JDUTIL01" || error("Not a JDUTIL01 matrix")
        version,ngoods,nbuyers,entries,seed,flags,reserved = Int.(read!(io,Vector{UInt64}(undef,7)))
        version==1 && flags==0 && reserved==0 || error("Unsupported JD utility format")
        ngoods>0 && nbuyers>0 && entries>0 || error("Empty JD matrix")
        m = users === nothing ? nbuyers : Int(users)
        0<m<=nbuyers || throw(ArgumentError("requested $m users, file contains $nbuyers"))
        filesize(path)==64+8*(nbuyers+1)+12entries+8nbuyers || error("Truncated or extended JD matrix file")
        colptr = Int.(read!(io,Vector{UInt64}(undef,m+1)))
        first(colptr)==1 && all(>(0),diff(colptr)) && last(colptr)<=entries+1 || error("Invalid CSC column pointers")
        count=last(colptr)-1
        seek(io,64+8*(nbuyers+1))
        rows=Int.(read!(io,Vector{UInt32}(undef,count)))
        all(j->1<=j<=ngoods,rows) || error("Invalid product index")
        seek(io,64+8*(nbuyers+1)+4entries)
        values=read!(io,Vector{Float64}(undef,count))
        all(x->isfinite(x)&&x>0&&isinteger(x),values) || error("Utilities must be positive raw click counts")
        seek(io,64+8*(nbuyers+1)+12entries)
        w=read!(io,Vector{Float64}(undef,m))
        all(x->isfinite(x)&&0<x<1,w) || error("Invalid raw budget draws")
        raw_budget_total=sum(w)
        w ./= raw_budget_total
        active=falses(ngoods);active[rows].=true
        goods=findall(active);mapping=cumsum(active)
        rows .= mapping[rows]
        n=length(goods)
        c=SparseMatrixCSC(n,m,colptr,rows,values)
        (;n,m,c,w,goods,seed,raw_budget_total,full_users=nbuyers,
          utility_normalization="none",budget_normalization="sum_one")
    end
end
end


m = 30
n = 20
ρ = 0.8
γ = 1e3
bool_run_conic = false
f0 = FisherMarket(m, n; ρ=ρ, bool_unit=true)
# endowments
b = rand(m, n)
b = b ./ sum(b; dims=1)

σ = f0.σ
c = f0.c
ρfmt = @sprintf("%+.1f", ρ)
σfmt = @sprintf("%+.1f", f0.σ)


ϕ(p) = begin
    return sum(p[i] for i in 1:n) +
           sum(f0.w[i] * log.(f0.w[i] * f0.f(p, i)^(1 / σ)) for i in 1:m)
end
g(p) = ForwardDiff.gradient(ϕ, p)
H(p) = ForwardDiff.hessian(ϕ, p)

fi(p, i) = begin
    return f0.f(p, i)
end
gi(p, i) = begin
    _f(p) = fi(p, i)
    ForwardDiff.gradient(_f, p)
end

xx(p) = begin
    _f = zeros(f0.m)
    _g = zeros(f0.m, f0.n)
    _x = zeros(f0.m, f0.n)
    for i in 1:f0.m
        _f[i], _g[i, :], _ = f0.f∇f(p, i)
        _x[i, :] = -_g[i, :] / _f[i] / f0.σ * p'b[i, :]
    end
    return _x
end
jx(p) = begin
    _f = zeros(f0.m)
    _g = zeros(f0.m, f0.n)
    _J = zeros(f0.n, f0.n)
    for i in 1:f0.m
        _f[i], _g[i, :], Hi = f0.f∇f(p, i)
        _J += p'b[i, :] / f0.σ / _f[i] * diagm(Hi) - p'b[i, :] / f0.σ / (_f[i]^2) * _g[i, :] * _g[i, :]' + _g[i, :] / _f[i] / f0.σ * b[i, :]'
    end
    return _J
end


xp(p) = begin
    return hcat([p'b[i, :] * (c[i, :] ./ p) .^ (1 + σ) / sum(p[j]^(-σ) * c[i, j]^(1 + σ) for j in 1:n) for i in 1:m]...)'
end
z(p) = -sum(xp(p); dims=1)[:] .+ 1

# known capital
xpw(p) = begin
    return hcat([f0.w[i] * (c[i, :] ./ p) .^ (1 + σ) / sum(p[j]^(-σ) * c[i, j]^(1 + σ) for j in 1:n) for i in 1:m]...)'
end
zpw(p) = -sum(xpw(p); dims=1)[:] .+ 1
Zpw(p) = ForwardDiff.jacobian(zpw, p)

# least square
ϵ(p) = sum(abs.(z(p)) .^ 2) / 2
Z(p) = ForwardDiff.jacobian(z, p)
gϵ(p) = ForwardDiff.gradient(ϵ, p)
Hϵ(p) = ForwardDiff.hessian(ϵ, p)

# potential reduction based on least square
Ψ(p) = begin
    return log(ϵ(p))
end
gΨ(p) = ForwardDiff.gradient(Ψ, p)
HΨ(p) = ForwardDiff.hessian(Ψ, p)


# approximate convex potential
μ = 1e-5
bb = sum(b; dims=1)[:]
kl(pw) = begin
    p = pw[1:n]
    w = pw[n+1:end]
    return sum(p[i] * log(p[i] / w[i]) for i in 1:n)
end
ϕw(pw) = begin
    p = pw[1:n]
    w = pw[n+1:end]
    return p'bb +
           sum(w[i] * log.(w[i] * f0.f(p, i)^(1 / σ)) for i in 1:m) +
           μ * sum(log(p' * b[i, :] - w[i]) for i in 1:m)
end
gw(pw) = ForwardDiff.gradient(ϕw, pw)
Hw(pw) = ForwardDiff.hessian(ϕw, pw)


bool_multiple = false
bool_square = false

if bool_multiple
    # Gjerstad's counter-example
    ρ = -1.1
    δ = 1 / (1 - ρ)
    a = 2 / (1 - 2 * δ)^(1 / δ)
    m = 2
    n = 2
    # elasticity of substitution is 1+σ = 0.25
    γ = 1e3
    bool_run_conic = false
    f0 = FisherMarket(m, n; ρ=ρ, bool_unit=true)
    f0.c = [a 1; 1 a]
    b = [1 0; 0 1]
    pₛ = [0.01900011819411108, 0.8739340528089753]
else
    m = 4
    n = 5
    # endowments
    if bool_square
        # ignore m
        b = I(n)
        m = n
    else
        b = rand(m, n)
        b = b ./ sum(b; dims=1)
    end
    ρ = 0.5
    γ = 1e3
    f0 = FisherMarket(m, n; ρ=ρ, bool_unit=true, sparsity=0.95)
end
bool_run_conic = false


σ = f0.σ
c = f0.c
ρfmt = @sprintf("%+.1f", ρ)
σfmt = @sprintf("%+.1f", f0.σ)


# ------------------------------------------------------
# functions for arrow-debreu
# ------------------------------------------------------
# compute x, the following should agree with each other
__xx(p) = begin
    _f = zeros(f0.m)
    _g = zeros(f0.m, f0.n)
    _x = zeros(f0.m, f0.n)
    for i in 1:f0.m
        _f[i], _g[i, :], _ = f0.f∇f(p, i)
        _x[i, :] = -_g[i, :] / _f[i] / f0.σ * p'b[i, :]
    end
    return _x
end
xp(p) = begin
    return hcat([p'b[i, :] * (c[i, :] ./ p) .^ (1 + σ) / sum(p[j]^(-σ) * c[i, j]^(1 + σ) for j in 1:n) for i in 1:m]...)'
end
xw(p) = begin
    return hcat([_w[i] * (c[i, :] ./ p) .^ (1 + σ) / sum(p[j]^(-σ) * c[i, j]^(1 + σ) for j in 1:n) for i in 1:m]...)'
end

# only compute i-th BR map
__x(p, i) = begin
    _f, _g, _ = f0.f∇f(p, i)
    _x = -_g / _f / f0.σ * p'b[i, :]
    return _x
end
# only compute i-th BR map, but in fisher case
__x(p, i) = begin
    _f, _g, _ = f0.f∇f(p, i)
    _x = -_g / _f / f0.σ * w[i]
    return _x
end
__Jx(p, i) = begin
    _xi(p) = __x(p, i)
    return ForwardDiff.jacobian(_xi, p)
end
__tJx(p, i) = begin
    _xi(p) = b[i, :] .* p - __x(p, i) .* p
    return ForwardDiff.jacobian(_xi, p)
end


# compute jacobian of sum(x)
jpsi(p) = begin
    _f = zeros(f0.m)
    _g = zeros(f0.m, f0.n)
    _J = zeros(f0.n, f0.n)
    for i in 1:f0.m
        _f[i], _g[i, :], Hi = f0.f∇f(p, i)
        _J += (
            -p'b[i, :] / f0.σ / _f[i] * diagm(Hi) +
            p'b[i, :] / f0.σ / (_f[i]^2) * _g[i, :] * _g[i, :]'
        )
    end
    return _J
end
jx(p) = begin
    _f = zeros(f0.m)
    _g = zeros(f0.m, f0.n)
    _J = zeros(f0.n, f0.n)
    for i in 1:f0.m
        _f[i], _g[i, :], Hi = f0.f∇f(p, i)
        _J += (
            -p'b[i, :] / f0.σ / _f[i] * diagm(Hi) +
            p'b[i, :] / f0.σ / (_f[i]^2) * _g[i, :] * _g[i, :]' -
            _g[i, :] / _f[i] / f0.σ * b[i, :]'
        )
    end
    return _J
end
__Xp(p) = ForwardDiff.jacobian((p) -> sum(xp(p); dims=1)[:], p)


# market excess
z(p) = -sum(xp(p); dims=1)[:] .+ 1
Z(p) = ForwardDiff.jacobian(z, p)
zw(p) = -sum(xw(p); dims=1)[:] .+ 1
Zw(p) = ForwardDiff.jacobian(zw, p)

# least square function to minimize z(p)
ϵ(p) = sum(abs.(z(p)) .^ 2) / 2
gϵ(p) = ForwardDiff.gradient(ϵ, p)
Hϵ(p) = ForwardDiff.hessian(ϵ, p)



# ------------------------------------------------------
# the following is for monotone transformation
# ------------------------------------------------------
# ρₛ = -1.0
ρₛ = -1.0
α = 1 - ρₛ

# transform d to p
dp(d) = begin
    p = d .^ α
    return p
end

# compute t(d), T(d)
t(d) = begin
    p = d .^ α
    return z(p) .* d
end

__tx(d, i) = begin
    p = d .^ α
    return __x(p, i) .* d
end

__tJxd(d, i) = begin
    _xi(d) = b[i, :] .* d - __tx(d, i)
    return ForwardDiff.jacobian(_xi, d)
end

T(d) = ForwardDiff.jacobian(t, d)

# analytical jacobian of sum(tilde x(d))
_txd(d) = begin
    p = d .^ α
    P = diagm(p)
    jacx = jx(p)
    xx = xp(p)
    xs = sum(xx; dims=1)[:]
    jacxx = α * (P .^ (1 / α)) * jacx * (P .^ (1 - 1 / α)) - diagm(xs) + I
    return jacxx
end
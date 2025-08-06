# Linear Systems

This page documents the linear system solvers and utilities in ExchangeMarket.jl.

## Linear System Solvers

The package includes efficient solvers for linear systems arising in market computations.

### Basic Linear System Solving

```julia
# Solve linear system Ax = b
A = rand(10, 10)
b = rand(10)
x = A \ b
```

## Krylov Methods

Krylov subspace methods for large-scale problems.

### Example

```julia
# Use iterative methods for large systems
using IterativeSolvers

# Solve with CG method
x = cg(A, b)
```

## DRQ Updates

Diagonal + Rank-q updates for efficient matrix operations.

### Example

```julia
# Efficient rank-1 updates
A = rand(100, 100)
u = rand(100)
v = rand(100)

# Sherman-Morrison update: A + uv'
A_updated = A + u * v'
```

## Matrix Operations

### Sherman-Morrison-Woodbury Updates

Efficient matrix updates for rank modifications.

```julia
# Sherman-Morrison formula
# (A + uv')^(-1) = A^(-1) - (A^(-1)u v'A^(-1))/(1 + v'A^(-1)u)

function sherman_morrison_update!(A_inv, u, v)
    w = A_inv * u
    z = A_inv' * v
    denominator = 1 + dot(v, w)
    A_inv .-= (w * z') / denominator
end
```

## Preconditioners

Preconditioning techniques for iterative solvers.

### Example

```julia
# Diagonal preconditioner
function diagonal_preconditioner(A)
    diag(A)
end

# Use with solver
precond = diagonal_preconditioner(A)
```

## Iterative Solvers

Iterative methods for large linear systems.

### Example

```julia
# Conjugate gradient method
function conjugate_gradient(A, b, tol=1e-6, max_iter=1000)
    n = length(b)
    x = zeros(n)
    r = copy(b)
    p = copy(r)
    
    for iter in 1:max_iter
        Ap = A * p
        alpha = dot(r, r) / dot(p, Ap)
        x .+= alpha * p
        r_new = r - alpha * Ap
        
        if norm(r_new) < tol
            break
        end
        
        beta = dot(r_new, r_new) / dot(r, r)
        p = r_new + beta * p
        r = r_new
    end
    
    return x
end
```

## Direct Solvers

Direct methods for small to medium systems.

### Example

```julia
# LU decomposition
function lu_solve(A, b)
    F = lu(A)
    return F \ b
end

# Cholesky decomposition (for symmetric positive definite)
function cholesky_solve(A, b)
    F = cholesky(A)
    return F \ b
end
```

## Utility Functions

### Condition Number

```julia
# Compute condition number
function condition_number(A)
    σ = svd(A).S
    return σ[1] / σ[end]
end
```

### Residual Norm

```julia
# Compute residual norm
function residual_norm(A, x, b)
    return norm(A * x - b)
end
```

## Performance Considerations

### Sparse Matrices

For large sparse systems, use sparse matrix formats:

```julia
using SparseArrays

# Create sparse matrix
A_sparse = sprand(1000, 1000, 0.01)
b = rand(1000)

# Solve sparse system
x = A_sparse \ b
```

### Memory Efficiency

For very large systems, consider:

1. **Iterative methods**: Use CG, GMRES, or BiCGSTAB
2. **Preconditioning**: Apply appropriate preconditioners
3. **Sparse storage**: Use compressed formats
4. **Block operations**: Exploit matrix structure

## Integration with ExchangeMarket.jl

Linear system solvers are used internally by the optimization algorithms:

- **HessianBar**: Uses direct solvers for Newton steps
- **PathFol**: Uses iterative solvers for path following
- **Tât**: Uses simple matrix operations for price updates
- **PropRes**: Uses basic linear algebra for proportional updates 
# Stable method identifiers and device-specific names for reports.
function linear_method_key(method)
    name = lowercase(string(method))
    name in ("pr", "pr (shared target)") && return :pr
    name == "pcg" && return :pcg
    name in ("perturbed", "factorization", "factorization (cpu)", "factorization (cudss)") && return :perturbed
    throw(ArgumentError("Unknown linear comparison method: $method"))
end
function linear_method_label(method, device)
    key = linear_method_key(method)
    key == :pr && return "PR"
    key == :pcg && return "PCG"
    string(device) == "cpu" && return "Factorization (CPU)"
    string(device) == "gpu" && return "Factorization (cuDSS)"
    throw(ArgumentError("Unknown device: $device"))
end
linear_method_family(method) = linear_method_key(method) == :perturbed ? "Factorization" : linear_method_label(method, :cpu)

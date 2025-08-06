using Documenter

makedocs(
    sitename="ExchangeMarket.jl",
    authors="Chuwen Zhang <chuwzhang@gmail.com>",
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => [
            "Models" => "api/models.md",
            "Algorithms" => "api/algorithms.md",
            "Linear Systems" => "api/linsys.md",
            "Utilities" => "api/utils.md",
        ],
        "Examples" => [
            "Fisher Market" => "examples/fisher_market.md",
        ],
        "Tutorials" => [
            "Basic Usage" => "tutorials/basic_usage.md",
        ],
    ],
    format=Documenter.HTML(),
    checkdocs=:none,
    doctest=false,
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() for more information.
deploydocs(
    repo="github.com/bzhangcw/ExchangeMarket.jl.git",
    devbranch="main",
)
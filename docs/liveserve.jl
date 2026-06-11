# Live-reload docs server.
#
#   julia --project=docs docs/liveserve.jl
#
# Watches docs/src and src/ (for docstrings), reruns make.jl on any
# change, and serves at http://localhost:8000.
using LiveServer
servedocs(; literate_dir = nothing, skip_dirs = [joinpath("docs", "build")])

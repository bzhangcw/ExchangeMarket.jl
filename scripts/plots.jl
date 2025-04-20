__backend = nothing
switch_to_pdf(; bool_use_html=false) = begin
    if !bool_use_html
        pgfplotsx()
        __backend = :pdf
    else
        plotly()
        __backend = :html
    end
end

function reset_size(n, R)
    cc.n = n
    cc.R = R
    cc.N = R * n
    @info """reset size to
    n: $(cc.n)
    R: $(cc.R)
    N: $(cc.N) 
    """
end

print_tex(io::IO, x::Char) = print(io, L"$x$")
function generate_empty(; use_html=__backend == :html, title="", shape=:wide, settick=true)
    if use_html
        switch_to_pdf(; bool_use_html=true)
    else
        switch_to_pdf(; bool_use_html=false)
    end
    return plot(
        extra_plot_kwargs=use_html ? Dict(
            :include_mathjax => "cdn",
        ) : Dict(),
        labelfontsize=20,
        xtickfont=font(16),
        ytickfont=font(16),
        legendfontsize=20,
        titlefontsize=20,
        yscale=settick ? :log10 : :auto,
        yticks=settick ? 10.0 .^ (-10:1:3) : :auto,
        legend=:topright,
        legendfonthalign=:left,
        title=title,
        linewidth=2,
        size=shape == :wide ? (900, 600) : (600, 500),
    )
end
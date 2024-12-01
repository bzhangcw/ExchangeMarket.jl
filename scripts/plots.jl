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

function generate_empty(; use_html=__backend == :html, title="", shape=:wide)
    return plot(
        extra_plot_kwargs=use_html ? Dict(
            :include_mathjax => "cdn",
        ) : Dict(),
        labelfontsize=20,
        xtickfont=font(15),
        ytickfont=font(15),
        legendfontsize=20,
        titlefontsize=20,
        yscale=:log10,
        yticks=10.0 .^ (-10:1:0),
        xlabel=L"$k$",
        ylabel="value",
        legend=:topright,
        legendfonthalign=:left,
        title=title,
        linewidth=2,
        size=shape == :wide ? (1000, 600) : (700, 600),
    )
end
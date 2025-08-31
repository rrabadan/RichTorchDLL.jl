using CairoMakie: Figure, Axis, hist!, stephist!, lines!, text!, axislegend
using CairoMakie: band!, barplot!, scatter!, hlines!
using StatsBase: fit, Histogram

function histogram_plot!(
    f,
    x;
    bins = 100,
    limits = (nothing, nothing),
    title = "",
    xlabel = "",
    ylabel = "Entries",
    histtype = :step,
    color = :blue,
    linewidth = 2,
)
    axis = Axis(
        f,
        limits = limits,
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        xticklabelsize = 18,
        yticklabelsize = 18,
        xlabelsize = 20,
        ylabelsize = 20,
        titlesize = 20,
    )
    if histtype == :step
        axis = stephist!(axis, x, bins = bins, linewidth = linewidth, color = color)
    else
        axis =
            hist!(axis, x, bins = bins, color = color, strokewidth = 1, strokecolor = color)
    end
    return axis
end

function histogram_plot(
    x;
    bins = 100,
    limits = (nothing, nothing),
    title = "",
    xlabel = "",
    ylabel = "Entries",
    histtype = :step,
    color = :blue,
    linewidth = 2,
    figsize = (600, 400),
)
    f = Figure(size = figsize)
    histogram_plot!(
        f[1, 1],
        x;
        bins = bins,
        limits = limits,
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        histtype = histtype,
        color = color,
        linewidth = linewidth,
    )
    return f
end

"""
    multi_histogram!(
        f,
        data_arrays;
        bins = 100,
        limits=(nothing, nothing),
        title = "",
        xlabel = "",
        ylabel = "Entries",
        labels = nothing,
        colors = nothing,
        normalization = :pdf,
        alpha = 0.7,
        grid = true,
        histtype = :step,
        show_legend = true,
        legend_position = :topright
    )

Plot multiple histograms on the same axis for comparison.

# Arguments
- `f`: Figure object to plot on
- `data_arrays`: Array of data arrays to plot histograms for
- `bins`: Number of bins or array of bin edges
- `limits`: Tuple specifying the (x, y) limits for the plot
- `title`: Title of the plot
- `xlabel`: X-axis label
- `ylabel`: Y-axis label
- `labels`: Array of labels for each data array (for legend)
- `colors`: Array of colors for each histogram
- `normalization`: Normalization method to apply to the histograms (:none, :pdf, :density, :probability)
- `alpha`: Transparency of the histograms
- `grid`: Whether to show grid lines
- `histtype`: Type of histogram (:step, :bar, :stepfilled)
- `show_legend`: Whether to show the legend
- `legend_position`: Position of the legend (:topright, :topleft, :bottomright, :bottomleft)

# Example
```julia
x1 = randn(1000)
x2 = randn(1000) .+ 2
fig = multi_histogram([x1, x2], 
    labels = ["Distribution 1", "Distribution 2"],
    title = "Comparison of Distributions",
    xlabel = "Value",
    normalize = true
)
```
"""
function multi_histogram!(
    f,
    data_arrays;
    bins = 100,
    limits = (nothing, nothing),
    title = "",
    xlabel = "",
    ylabel = "Entries",
    labels = nothing,
    colors = nothing,
    normalization = :pdf,
    alpha = 0.7,
    grid = true,
    histtype = :step,
    show_legend = true,
    legend_position = :rt,
)
    n_datasets = length(data_arrays)

    # Set default labels if not provided
    if isnothing(labels)
        labels = ["Dataset $i" for i = 1:n_datasets]
    end

    # Set default colors if not provided
    if isnothing(colors)
        # Use a colormap to generate distinct colors
        colormap = cgrad(:Dark2_8, n_datasets, categorical = true)
        colors = [colormap[i] for i = 1:n_datasets]
    end

    ax = Axis(
        f,
        limits = limits,
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        xticklabelsize = 14,
        yticklabelsize = 14,
        xlabelsize = 16,
        ylabelsize = 16,
        titlesize = 18,
    )

    if grid
        ax.xgridvisible = true
        ax.ygridvisible = true
    end

    hist_objects = []

    for (i, data) in enumerate(data_arrays)
        # Filter out NaN and Inf values
        valid_data = filter(x -> !isnan(x) && !isinf(x), data)

        if histtype == :step
            hist_obj = stephist!(
                ax,
                valid_data,
                bins = bins,
                color = colors[i],
                linewidth = 2,
                label = labels[i],
                normalization = normalization,
            )
            push!(hist_objects, hist_obj)
        else
            # For bar histograms, use the built-in hist! function
            hist_obj = hist!(
                ax,
                valid_data,
                bins = bins,
                color = (colors[i], alpha),
                strokewidth = 1,
                strokecolor = colors[i],
                label = labels[i],
                normalization = normalization,
            )
            push!(hist_objects, hist_obj)
        end
    end
    if show_legend
        axislegend(
            ax,
            position = legend_position,
            fontsize = 12,
            framecolor = :black,
            framealpha = 0.1,
        )
    end
    return ax
end

function multi_histogram(
    data_arrays;
    bins = 100,
    limits = (nothing, nothing),
    title = "",
    xlabel = "",
    ylabel = "Entries",
    labels = nothing,
    colors = nothing,
    normalization = :pdf,
    alpha = 0.7,
    grid = true,
    histtype = :step,
    show_legend = true,
    legend_position = :rt,
    figsize = (600, 400),
)
    f = Figure(size = figsize)
    multi_histogram!(
        f[1, 1],
        data_arrays;
        bins = bins,
        limits = limits,
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        labels = labels,
        colors = colors,
        normalization = normalization,
        alpha = alpha,
        grid = true,
        histtype = histtype,
        show_legend = show_legend,
        legend_position = legend_position,
    )
    return f
end

"""
    histogram_with_ratio(
        data1, 
        data2; 
        label1 = "Data 1", 
        label2 = "Data 2",
        title = "Histogram Comparison",
        xlabel = "Value",
        ylabel = "Entries",
        ratio_ylabel = "Ratio",
        bins = 100,
        color1 = :royalblue,
        color2 = :crimson,
        figsize = (800, 600),
        ratio_range = (0.5, 1.5),
        normalize = false,
        hist_type = :step
    )

Create a histogram comparison with a ratio panel below.

# Arguments
- `data1`: First data array
- `data2`: Second data array
- `label1`: Legend label for first dataset
- `label2`: Legend label for second dataset
- `title`: Plot title
- `xlabel`: X-axis label
- `ylabel`: Y-axis label for histogram
- `ratio_ylabel`: Y-axis label for ratio panel
- `bins`: Number of bins or array of bin edges
- `color1`: Color for first histogram
- `color2`: Color for second histogram
- `figsize`: Figure size
- `ratio_range`: Y-axis limits for ratio panel
- `normalize`: If true, normalize histograms to have the same area
- `hist_type`: Type of histogram (:step, :bar, :stepfilled)

# Returns
- `Figure` object that can be displayed or saved

# Example
```julia
x1 = randn(1000)
x2 = randn(1200) .* 1.1 .+ 0.2
fig = histogram_with_ratio(
    x1, x2,
    label1 = "Simulation",
    label2 = "Data",
    title = "Energy Distribution",
    xlabel = "Energy [GeV]",
    normalize = true
)
```
"""
function histogram_with_ratio(
    data1,
    data2;
    label1 = "Data 1",
    label2 = "Data 2",
    title = "Histogram Comparison",
    xlabel = "Value",
    ylabel = "Entries",
    ratio_ylabel = "Ratio",
    bins = 100,
    color1 = :royalblue,
    color2 = :crimson,
    figsize = (800, 600),
    ratio_range = (0.5, 1.5),
    normalize = false,
    hist_type = :step,
)
    # Filter out NaN and Inf values
    valid_data1 = filter(x -> !isnan(x) && !isinf(x), data1)
    valid_data2 = filter(x -> !isnan(x) && !isinf(x), data2)

    # Create common binning for both histograms
    x_min = min(minimum(valid_data1), minimum(valid_data2))
    x_max = max(maximum(valid_data1), maximum(valid_data2))

    # Add a small margin
    margin = 0.05 * (x_max - x_min)
    x_min -= margin
    x_max += margin

    if typeof(bins) <: Integer
        bin_edges = range(x_min, x_max, length = bins + 1)
    else
        bin_edges = bins
    end

    # Create histograms with the same binning
    h1 = fit(Histogram, valid_data1, bin_edges)
    h2 = fit(Histogram, valid_data2, bin_edges)

    counts1 = h1.weights
    counts2 = h2.weights
    edges = h1.edges[1]
    centers = [(edges[i] + edges[i+1]) / 2 for i = 1:length(edges)-1]

    # Normalize if requested
    if normalize
        sum1 = sum(counts1)
        sum2 = sum(counts2)
        if sum1 > 0
            counts1 = counts1 ./ sum1
        end
        if sum2 > 0
            counts2 = counts2 ./ sum2
        end
    end

    # Calculate ratio
    ratio = similar(counts1)
    for i = 1:length(counts1)
        if counts2[i] > 0
            ratio[i] = counts1[i] / counts2[i]
        else
            ratio[i] = NaN  # Avoid division by zero
        end
    end

    # Create figure with grid layout - top for histograms, bottom for ratio
    fig = Figure(size = figsize)

    # Create the main histogram panel
    ax_hist = Axis(fig[1, 1], title = title, ylabel = ylabel)

    # Hide x-axis labels on top panel
    hidexdecorations!(ax_hist, grid = false, ticks = false)

    # Create the ratio panel
    ax_ratio = Axis(fig[2, 1], xlabel = xlabel, ylabel = ratio_ylabel)

    # Create histograms based on type
    if hist_type == :step || hist_type == :stepfilled
        # Use Makie's built-in stephist! for step and stepfilled
        stephist!(
            ax_hist,
            valid_data1,
            bins = bin_edges,
            color = color1,
            linewidth = 2,
            label = label1,
            normalization = normalize ? :pdf : :none,
        )
        stephist!(
            ax_hist,
            valid_data2,
            bins = bin_edges,
            color = color2,
            linewidth = 2,
            label = label2,
            normalization = normalize ? :pdf : :none,
        )
        if hist_type == :stepfilled
            # Overlay filled bands for stepfilled
            h1 = fit(Histogram, valid_data1, bin_edges)
            h2 = fit(Histogram, valid_data2, bin_edges)
            edges = h1.edges[1]
            counts1 =
                normalize && sum(h1.weights) > 0 ? h1.weights ./ sum(h1.weights) :
                h1.weights
            counts2 =
                normalize && sum(h2.weights) > 0 ? h2.weights ./ sum(h2.weights) :
                h2.weights
            x_steps = [edges[i] for i = 1:length(edges) for _ = 1:2][1:end-1]
            y_steps1 = [v for v in counts1 for _ = 1:2]
            y_steps2 = [v for v in counts2 for _ = 1:2]
            band!(ax_hist, x_steps, zeros(length(x_steps)), y_steps1, color = (color1, 0.3))
            band!(ax_hist, x_steps, zeros(length(x_steps)), y_steps2, color = (color2, 0.3))
        end
    else
        # For bar histograms
        barwidth = (edges[2] - edges[1]) * 0.4  # Adjust width for side-by-side bars

        # Plot histograms as bars
        barplot!(
            ax_hist,
            centers .- barwidth / 2,
            counts1,
            width = barwidth,
            color = (color1, 0.7),
            strokewidth = 1,
            strokecolor = color1,
            label = label1,
        )

        barplot!(
            ax_hist,
            centers .+ barwidth / 2,
            counts2,
            width = barwidth,
            color = (color2, 0.7),
            strokewidth = 1,
            strokecolor = color2,
            label = label2,
        )
    end

    # Plot ratio points
    scatter!(ax_ratio, centers, ratio, color = :black, markersize = 4)

    # Add connecting lines for ratio
    valid_idx = .!isnan.(ratio)
    lines!(ax_ratio, centers[valid_idx], ratio[valid_idx], color = :black, linewidth = 1)

    # Add a horizontal line at ratio=1
    hlines!(ax_ratio, [1.0], color = :gray, linestyle = :dash)

    # Set y-axis limits for ratio panel
    ax_ratio.limits = (nothing, nothing, ratio_range...)

    # Link x-axes
    linkxaxes!(ax_hist, ax_ratio)

    # Add legend to the histogram panel
    Legend(fig[1, 2], ax_hist, "Datasets", margin = (10, 10, 10, 10), labelsize = 14)

    # Set figure layout - give more space to histogram than ratio
    rowsize!(fig.layout, 1, 0.7)
    rowsize!(fig.layout, 2, 0.3)

    return fig
end


"""
    histogram_grid(
        data_arrays;
        titles = nothing,
        xlabels = nothing,
        ylabels = nothing,
        bins = 100,
        colors = nothing,
        grid_layout = nothing,
        figsize = (1000, 800),
        normalize = false,
        hist_type = :bar,
        link_axes = true,
        global_title = nothing
    )

Create a grid of histograms for multiple datasets.

# Arguments
- `data_arrays`: Array of data arrays to plot histograms for
- `titles`: Array of titles for each subplot
- `xlabels`: Array of x-axis labels for each subplot
- `ylabels`: Array of y-axis labels for each subplot
- `bins`: Number of bins or array of bin edges (can be an array of arrays for different bins per plot)
- `colors`: Array of colors for each histogram
- `grid_layout`: Tuple of (rows, cols) for the grid layout (calculated automatically if not provided)
- `figsize`: Size of the figure in pixels
- `normalize`: If true, normalize histograms to have unit area
- `hist_type`: Type of histogram (:bar, :step, :stepfilled)
- `link_axes`: If true, link x and y axes across all subplots
- `global_title`: Optional title for the entire figure

# Returns
- `Figure` object that can be displayed or saved

# Example
```julia
x1 = randn(1000)
x2 = randn(1000) .* 0.5
x3 = randn(1000) .+ 2
x4 = rand(1000) .* 5

fig = histogram_grid(
    [x1, x2, x3, x4],
    titles = ["Normal", "Narrow Normal", "Shifted Normal", "Uniform"],
    xlabels = ["Value" for _ in 1:4],
    colors = [:royalblue, :crimson, :darkgreen, :purple],
    global_title = "Distribution Comparison"
)
```
"""
function histogram_grid(
    data_arrays;
    titles = nothing,
    xlabels = nothing,
    ylabels = nothing,
    bins = 100,
    colors = nothing,
    grid_layout = nothing,
    figsize = (1000, 800),
    normalize = false,
    hist_type = :bar,
    link_axes = true,
    global_title = nothing,
)
    n_plots = length(data_arrays)

    # Determine grid layout if not provided
    if isnothing(grid_layout)
        n_cols = Int(ceil(sqrt(n_plots)))
        n_rows = Int(ceil(n_plots / n_cols))
        grid_layout = (n_rows, n_cols)
    else
        n_rows, n_cols = grid_layout
    end

    # Set default titles, labels if not provided
    if isnothing(titles)
        titles = ["Histogram $i" for i = 1:n_plots]
    end

    if isnothing(xlabels)
        xlabels = ["Value" for i = 1:n_plots]
    end

    if isnothing(ylabels)
        ylabels = ["Frequency" for i = 1:n_plots]
    end

    # Set default colors if not provided
    if isnothing(colors)
        colormap = cgrad(:Dark2_8, n_plots, categorical = true)
        colors = [colormap[i] for i = 1:n_plots]
    end

    # Create figure
    fig = Figure(size = figsize)

    # Add global title if provided
    if !isnothing(global_title)
        Label(fig[0, 1:n_cols], global_title, fontsize = 24)
    end

    axes = []
    extrema_values = []

    # Create subplots
    for i = 1:n_plots
        row = div(i - 1, n_cols) + 1
        col = mod(i - 1, n_cols) + 1

        ax = Axis(
            fig[row, col],
            title = titles[i],
            xlabel = xlabels[i],
            ylabel = ylabels[i],
            titlesize = 16,
            xlabelsize = 14,
            ylabelsize = 14,
        )

        push!(axes, ax)

        # Filter out NaN and Inf values
        valid_data = filter(x -> !isnan(x) && !isinf(x), data_arrays[i])

        # Store extrema for axis alignment
        push!(extrema_values, extrema(valid_data))

        # Create the histogram
        if hist_type == :step || hist_type == :stepfilled
            # Use Makie's built-in stephist! for step and stepfilled
            bin_setting = isa(bins, Array) && length(bins) == n_plots ? bins[i] : bins
            normalization_mode = normalize ? :pdf : :none
            stephist!(
                ax,
                valid_data,
                bins = bin_setting,
                color = colors[i],
                linewidth = 2,
                normalization = normalization_mode,
            )
            if hist_type == :stepfilled
                # Overlay filled band for stepfilled
                h = fit(Histogram, valid_data, nbins = bin_setting)
                edges = h.edges[1]
                counts =
                    normalization_mode == :pdf && sum(h.weights) > 0 ?
                    h.weights ./ sum(h.weights) : h.weights
                x_steps = [edges[j] for j = 1:length(edges) for _ = 1:2][1:end-1]
                y_steps = [v for v in counts for _ = 1:2]
                band!(
                    ax,
                    x_steps,
                    zeros(length(x_steps)),
                    y_steps,
                    color = (colors[i], 0.3),
                )
            end
        else
            # For bar histograms, use the built-in hist! function
            bin_setting = isa(bins, Array) && length(bins) == n_plots ? bins[i] : bins
            hist!(
                ax,
                valid_data,
                bins = bin_setting,
                color = (colors[i], 0.7),
                strokewidth = 1,
                strokecolor = colors[i],
                normalization = normalize ? :pdf : :none,
            )
        end
    end

    # Link axes if requested
    if link_axes && length(axes) > 1
        # Find common range for x-axis
        if link_axes
            x_min = minimum([ex[1] for ex in extrema_values])
            x_max = maximum([ex[2] for ex in extrema_values])

            for ax in axes
                ax.limits = (x_min, x_max, nothing, nothing)
            end

            linkyaxes!(axes...)
        end
    end

    return fig
end

"""
    plot_efficiency_vs_momentum(
        scores, 
        labels, 
        momentum, 
        threshold, 
        bin_edges;
        title = "Efficiency vs Momentum",
        xlabel = "Momentum",
        ylabel = "Efficiency",
        figsize = (600, 400),
        color = :royalblue,
        min_samples = 10
    )

Plot the efficiency as a function of momentum with error bars.

# Arguments
- `scores`: Vector of classifier scores
- `labels`: Vector of true labels (0 = negative, 1 = positive)
- `momentum`: Vector of momentum values for each sample
- `threshold`: Classification threshold (predict positive if score > threshold)
- `bin_edges`: Vector of bin edges for momentum bins
- `title`: Plot title
- `xlabel`: X-axis label
- `ylabel`: Y-axis label
- `figsize`: Figure size in pixels
- `color`: Color for the plot points and line
- `min_samples`: Minimum number of positive samples required in a bin for plotting

# Returns
A tuple containing:
- `fig`: The Figure object
- `efficiency_data`: The calculated efficiency data
"""
function plot_efficiency_vs_momentum(
    scores::AbstractVector{<:Real},
    labels::AbstractVector{<:Integer},
    momentum::AbstractVector{<:Real},
    threshold::Real,
    bin_edges::AbstractVector{<:Real};
    title = "Efficiency vs Momentum",
    xlabel = "Momentum",
    ylabel = "Efficiency",
    figsize = (600, 400),
    color = :royalblue,
    min_samples = 1,
    show_bin_edges = true,
    tick_format = x -> string(Int(round(x))),
)
    # Calculate efficiency per bin
    eff_data = efficiency_per_momentum_bin(scores, labels, momentum, threshold, bin_edges)

    # Create figure
    fig = Figure(size = figsize)
    ax = Axis(
        fig[1, 1],
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        titlesize = 20,
        xlabelsize = 16,
        ylabelsize = 16,
        xticklabelsize = 14,
        yticklabelsize = 14,
    )

    # Filter out bins with too few samples
    valid_bins = eff_data.n_pos .>= min_samples
    x_values = eff_data.bin_centers[valid_bins]
    y_values = eff_data.efficiency[valid_bins]
    errors = eff_data.efficiency_error[valid_bins]

    # Plot efficiency points with error bars
    errorbars!(ax, x_values, y_values, errors, color = color, whiskerwidth = 10)
    scatter!(ax, x_values, y_values, color = color, markersize = 8)

    # Connect points with lines where data exists
    valid_mask = .!isnan.(y_values)
    if any(valid_mask)
        lines!(ax, x_values[valid_mask], y_values[valid_mask], color = color, linewidth = 2)
    end

    # Add grid lines
    ax.xgridvisible = true
    ax.ygridvisible = true

    # Set y-axis limits from 0 to 1 for efficiency
    ax.limits = (nothing, nothing, 0, 1.05)

    # Set y-axis ticks to 0, 0.2, 0.4, 0.6, 0.8, 1.0
    ax.yticks = 0:0.1:1.0

    # Set x-axis ticks based on bin edges if requested
    if show_bin_edges
        ax.xticks = (bin_edges, map(tick_format, bin_edges))
    end

    # Add text with threshold information
    #text!(
    #    ax,
    #    "Threshold: $(round(threshold, digits=3))",
    #    position = (mean(ax.limits[1:2]), 0.05),
    #    align = (:center, :bottom),
    #    fontsize = 12,
    #)

    return (figure = fig, efficiency_data = eff_data)
end

"""
    plot_bin_efficiency_data(
        bin_centers::AbstractVector{<:Real},
        efficiency::AbstractVector{<:Real},
        efficiency_error::AbstractVector{<:Real},
        bin_edges::AbstractVector{<:Real};
        achieved_misid = nothing,
        misid_error = nothing,
        title = "Efficiency vs Momentum",
        xlabel = "Momentum",
        ylabel = "Efficiency",
        figsize = (600, 400),
        color = :royalblue,
        min_samples = 10,  # For API compatibility
        show_bin_edges = true,
        tick_format = x -> string(Int(round(x))),
        show_misid = false
    )

Plot efficiency vs momentum data directly from provided arrays.
This function allows plotting efficiency data calculated with any method,
including bin-by-bin misID rate optimization.

# Arguments
- `bin_centers`: Centers of momentum bins
- `efficiency`: Efficiency in each bin
- `efficiency_error`: Error on the efficiency in each bin
- `bin_edges`: Edges of momentum bins (for x-axis ticks)
- `achieved_misid`: Optional array of achieved misID rates in each bin
- `misid_error`: Optional array of errors on the misID rates
- `title`: Plot title
- `xlabel`: X-axis label
- `ylabel`: Y-axis label
- `figsize`: Figure size in pixels
- `color`: Color for the points and lines
- `min_samples`: Not used in this function, but included for API compatibility
- `show_bin_edges`: Whether to show bin edges as x-axis ticks
- `tick_format`: Function to format tick labels
- `show_misid`: Whether to show achieved misID rate as a second y-axis

# Returns
A tuple containing:
- `figure`: The Figure object
- `ax`: The main Axis object
"""
function plot_bin_efficiency_data(
    bin_centers::AbstractVector{<:Real},
    efficiency::AbstractVector{<:Real},
    efficiency_error::AbstractVector{<:Real},
    bin_edges::AbstractVector{<:Real};
    achieved_misid = nothing,
    misid_error = nothing,
    title = "Efficiency vs Momentum",
    xlabel = "Momentum",
    ylabel = "Efficiency",
    figsize = (600, 400),
    color = :royalblue,
    show_bin_edges = true,
    tick_format = x -> string(Int(round(x))),
    show_misid = false,
    legend_position = :rt,
)
    # Create figure
    fig = Figure(size = figsize)
    ax = Axis(
        fig[1, 1],
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        titlesize = 20,
        xlabelsize = 16,
        ylabelsize = 16,
        xticklabelsize = 14,
        yticklabelsize = 14,
    )

    # Filter out NaN values
    valid_mask = .!isnan.(efficiency)
    x_values = bin_centers[valid_mask]
    y_values = efficiency[valid_mask]
    errors = efficiency_error[valid_mask]

    # Plot efficiency points with error bars
    errorbars!(ax, x_values, y_values, errors, color = color, whiskerwidth = 10)
    scatter!(ax, x_values, y_values, color = color, markersize = 8, label = "Efficiency")

    # Connect points with lines
    if any(valid_mask)
        lines!(ax, x_values, y_values, color = color, linewidth = 2)
    end

    # Add grid lines
    ax.xgridvisible = true
    ax.ygridvisible = true

    # Set y-axis limits from 0 to 1 for efficiency
    ax.limits = (nothing, nothing, 0, 1.05)

    # Set y-axis ticks to 0, 0.2, 0.4, 0.6, 0.8, 1.0
    ax.yticks = 0:0.1:1.0

    # Set x-axis ticks based on bin edges if requested
    if show_bin_edges
        ax.xticks = (bin_edges, map(tick_format, bin_edges))
    end

    # Add second y-axis for misID rate if requested
    if show_misid && achieved_misid !== nothing
        #ax2 = Axis(
        #    fig[1, 1],
        #    ylabel="MisID Rate",
        #    yaxisposition=:right,
        #    ytickformat=values -> ["$(round(v*100, digits=1))%" for v in values],
        #    ylabelsize=16,
        #    yticklabelsize=14,
        #)

        # Hide all elements except the y-axis
        #hidespines!(ax2)
        #hidexdecorations!(ax2)

        # Filter out NaN values for misID
        valid_misid_mask = .!isnan.(achieved_misid)
        misid_x = bin_centers[valid_misid_mask]
        misid_y = achieved_misid[valid_misid_mask]

        ax.ylabel = "Efficiency (MisID Rate)"

        # Plot misID rates
        misid_color = :darkred
        if misid_error !== nothing
            misid_err = misid_error[valid_misid_mask]
            errorbars!(
                ax,
                misid_x,
                misid_y,
                misid_err,
                color = misid_color,
                whiskerwidth = 10,
            )
        end

        scatter!(
            ax,
            misid_x,
            misid_y,
            color = misid_color,
            marker = :diamond,
            markersize = 8,
            label = "MisID Rate",
        )
        lines!(ax, misid_x, misid_y, color = misid_color, linewidth = 2, linestyle = :dash)

        # Set y-axis limits for misID rate (adjust as needed)
        # target_misid = mean(filter(!isnan, achieved_misid))
        # y_margin = max(0.02, 3 * std(filter(!isnan, achieved_misid)))
        # y_min = max(0, target_misid - y_margin)
        # y_max = min(1, target_misid + y_margin)
        # ax2.limits = (nothing, nothing, y_min, y_max)

        # Create a legend entry for misID rate
        # lines!(ax, [NaN], [NaN], color=misid_color, linestyle=:dash,
        #    linewidth=2)
        # scatter!(ax, [NaN], [NaN], color=misid_color, marker=:diamond,
        #    label="MisID Rate", markersize=8)

        axislegend(
            ax,
            position = legend_position,
            fontsize = 20,
            framecolor = :black,
            framealpha = 0.1,
        )
    end

    return (figure = fig, ax = ax)
end

"""
    compare_bin_efficiency_data(
        bin_centers_list::Vector{<:AbstractVector{<:Real}},
        efficiency_list::Vector{<:AbstractVector{<:Real}},
        efficiency_error_list::Vector{<:AbstractVector{<:Real}},
        bin_edges::AbstractVector{<:Real};
        labels = nothing,
        title = "Efficiency Comparison",
        xlabel = "Momentum",
        ylabel = "Efficiency",
        figsize = (600, 400),
        colors = nothing,
        show_bin_edges = true,
        tick_format = x -> string(Int(round(x))),
        legend_position = :rb
    )

Compare multiple efficiency vs momentum data sets.
This function allows comparing efficiency data calculated with any method,
including bin-by-bin misID rate optimization.

# Arguments
- `bin_centers_list`: List of bin center vectors for each data set
- `efficiency_list`: List of efficiency vectors for each data set
- `efficiency_error_list`: List of efficiency error vectors for each data set
- `bin_edges`: Vector of bin edges for momentum bins (for x-axis ticks)
- `labels`: Optional list of labels for the legend
- `title`: Plot title
- `xlabel`: X-axis label
- `ylabel`: Y-axis label
- `figsize`: Figure size in pixels
- `colors`: Optional list of colors for each data set
- `show_bin_edges`: Whether to show bin edges as x-axis ticks
- `tick_format`: Function to format tick labels
- `legend_position`: Position of the legend

# Returns
A tuple containing:
- `figure`: The Figure object
- `ax`: The main Axis object
"""
function compare_bin_efficiency_data(
    bin_centers_list::Vector{<:AbstractVector{<:Real}},
    efficiency_list::Vector{<:AbstractVector{<:Real}},
    efficiency_error_list::Vector{<:AbstractVector{<:Real}},
    bin_edges::AbstractVector{<:Real};
    labels = nothing,
    title = "Efficiency Comparison",
    xlabel = "Momentum",
    ylabel = "Efficiency",
    figsize = (600, 400),
    colors = nothing,
    show_bin_edges = true,
    tick_format = x -> string(Int(round(x))),
    legend_position = :rb,
)
    n_datasets = length(bin_centers_list)
    @assert length(efficiency_list) == n_datasets "Must provide same number of efficiency vectors as bin center vectors"
    @assert length(efficiency_error_list) == n_datasets "Must provide same number of error vectors as bin center vectors"

    # Default labels if not provided
    if isnothing(labels)
        labels = ["Dataset $i" for i = 1:n_datasets]
    end

    # Default colors if not provided
    if isnothing(colors)
        colormap = cgrad(:Dark2_8, n_datasets, categorical = true)
        colors = [colormap[i] for i = 1:n_datasets]
    end

    # Create figure
    fig = Figure(size = figsize)
    ax = Axis(
        fig[1, 1],
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        titlesize = 24,  # Increased from 20
        xlabelsize = 20,  # Increased from 16
        ylabelsize = 20,  # Increased from 16
        xticklabelsize = 16,  # Increased from 14
        yticklabelsize = 16,  # Increased from 14
    )

    # Plot each dataset
    for i = 1:n_datasets
        bin_centers = bin_centers_list[i]
        efficiency = efficiency_list[i]
        errors = efficiency_error_list[i]

        # Filter out NaN values
        valid_mask = .!isnan.(efficiency)
        x_values = bin_centers[valid_mask]
        y_values = efficiency[valid_mask]
        error_values = errors[valid_mask]

        # Plot efficiency points with error bars
        errorbars!(
            ax,
            x_values,
            y_values,
            error_values,
            color = colors[i],
            whiskerwidth = 10,
        )
        scatter!(
            ax,
            x_values,
            y_values,
            color = colors[i],
            markersize = 8,
            label = labels[i],
        )

        # Connect points with lines
        if any(valid_mask)
            lines!(ax, x_values, y_values, color = colors[i], linewidth = 2)
        end
    end

    # Add grid lines
    ax.xgridvisible = true
    ax.ygridvisible = true

    # Set y-axis limits from 0 to 1 for efficiency
    ax.limits = (nothing, nothing, 0, 1.05)

    # Set y-axis ticks to 0, 0.2, 0.4, 0.6, 0.8, 1.0
    ax.yticks = 0:0.1:1.0

    # Set x-axis ticks based on bin edges if requested
    if show_bin_edges
        ax.xticks = (bin_edges, map(tick_format, bin_edges))
    end

    # Add legend
    axislegend(
        ax,
        position = legend_position,
        fontsize = 20,
        framecolor = :black,
        framealpha = 0.1,
    )

    return (figure = fig, ax = ax)
end

"""
    efficiency_vs_momentum_for_misid_rate(
        scores, 
        labels, 
        momentum, 
        misid_rate::Real,
        bin_edges;
        kwargs...
    )

Calculate the threshold for a target misidentification rate and plot the efficiency vs momentum.

# Arguments
- `scores`: Vector of classifier scores
- `labels`: Vector of true labels (0 = negative, 1 = positive)
- `momentum`: Vector of momentum values for each sample
- `misid_rate`: Target misidentification rate (e.g., 0.01 for 1%)
- `bin_edges`: Vector of bin edges for momentum bins
- Additional keyword arguments are passed to `plot_efficiency_vs_momentum`

# Returns
A tuple containing:
- `fig`: The Figure object
- `efficiency_data`: The calculated efficiency data
- `workingpoint`: Information about the chosen threshold
"""
function efficiency_vs_momentum_for_misid_rate(
    scores::AbstractVector{<:Real},
    labels::AbstractVector{<:Integer},
    momentum::AbstractVector{<:Real},
    misid_rate::Real,
    bin_edges::AbstractVector{<:Real};
    kwargs...,
)
    # Find threshold for given misid rate
    workingpoints = find_workingpoints(scores, labels, misid_rates = [misid_rate])
    wp = workingpoints[misid_rate]
    threshold = wp.threshold

    # Add misid rate to plot title
    title = get(kwargs, :title, "Efficiency vs Momentum")
    title = "$title (Misid Rate: $(100*misid_rate)%)"

    # Plot with the calculated threshold
    result = plot_efficiency_vs_momentum(
        scores,
        labels,
        momentum,
        threshold,
        bin_edges;
        title = title,
        kwargs...,
    )

    return (
        figure = result.figure,
        efficiency_data = result.efficiency_data,
        workingpoint = wp,
    )
end

"""
    efficiency_vs_momentum_with_per_bin_misid(
        scores::AbstractVector{<:Real}, 
        labels::AbstractVector{<:Integer}, 
        momentum::AbstractVector{<:Real}, 
        misid_rate::Real,
        bin_edges::AbstractVector{<:Real};
        title = "Efficiency vs Momentum (Per-Bin MisID)",
        xlabel = "Momentum",
        ylabel = "Efficiency",
        figsize = (600, 400),
        color = :royalblue,
        min_bin_samples = 10,
        show_bin_edges = true,
        tick_format = x -> string(Int(round(x))),
        show_misid = true
    )

Calculate and plot efficiency vs momentum with a per-bin misID rate optimization.
Unlike the global approach that uses a single threshold, this function
calculates a separate threshold for each momentum bin to achieve the target
misID rate within that bin.

# Arguments
- `scores`: Vector of classifier scores
- `labels`: Vector of true labels (0 = negative, 1 = positive)
- `momentum`: Vector of momentum values
- `misid_rate`: Target misidentification rate (e.g., 0.01 for 1%)
- `bin_edges`: Vector of bin edges for momentum bins
- `title`: Plot title
- `xlabel`: X-axis label
- `ylabel`: Y-axis label
- `figsize`: Figure size in pixels
- `color`: Color for the points and lines
- `min_bin_samples`: Minimum number of samples required in a bin for calculation
- `show_bin_edges`: Whether to show bin edges as x-axis ticks
- `tick_format`: Function to format tick labels
- `show_misid`: Whether to show achieved misID rate as a second y-axis

# Returns
A tuple containing:
- `figure`: The Figure object
- `bin_data`: The calculated bin-by-bin efficiency and misID data
"""
function efficiency_vs_momentum_with_per_bin_misid(
    scores::AbstractVector{<:Real},
    labels::AbstractVector{<:Integer},
    momentum::AbstractVector{<:Real},
    misid_rate::Real,
    bin_edges::AbstractVector{<:Real};
    title = "Efficiency vs Momentum (Per-Bin MisID)",
    xlabel = "Momentum",
    ylabel = "Efficiency",
    figsize = (600, 400),
    color = :royalblue,
    min_bin_samples = 10,
    show_bin_edges = true,
    tick_format = x -> string(Int(round(x))),
    show_misid = true,
    legend_position = :rc,
)
    # Calculate efficiency at target misID rate per bin
    bin_data = efficiency_per_momentum_bin_at_misid_rate(
        scores,
        labels,
        momentum,
        misid_rate,
        bin_edges;
        min_bin_samples = min_bin_samples,
    )

    # Update title to include misID rate
    title = "$title (Target MisID: $(100*misid_rate)%)"

    # Create the plot
    result = plot_bin_efficiency_data(
        bin_data.bin_centers,
        bin_data.efficiency,
        bin_data.efficiency_error,
        bin_edges;
        achieved_misid = bin_data.achieved_misid,
        misid_error = bin_data.misid_error,
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        figsize = figsize,
        color = color,
        show_bin_edges = show_bin_edges,
        tick_format = tick_format,
        show_misid = show_misid,
        legend_position = legend_position,
    )

    return (figure = result.figure, bin_data = bin_data)
end

"""
    compare_efficiency_vs_momentum(
        scores_list,
        labels_list,
        momentum_list,
        thresholds,
        bin_edges;
        labels = nothing,
        title = "Efficiency Comparison",
        xlabel = "Momentum",
        ylabel = "Efficiency",
        figsize = (600, 400),
        colors = nothing,
        min_samples = 10,
        legend_position = :rb
    )

Compare efficiency vs momentum curves for multiple classifiers or configurations.

# Arguments
- `scores_list`: List of score vectors for each classifier
- `labels_list`: List of label vectors for each classifier
- `momentum_list`: List of momentum vectors for each classifier
- `thresholds`: List of threshold values for each classifier
- `bin_edges`: Vector of bin edges for momentum bins
- `labels`: Optional list of labels for the legend
- `title`: Plot title
- `xlabel`: X-axis label
- `ylabel`: Y-axis label
- `figsize`: Figure size in pixels
- `colors`: Optional list of colors for each classifier
- `min_samples`: Minimum number of positive samples required in a bin for plotting
- `legend_position`: Position of the legend

# Returns
A tuple containing:
- `fig`: The Figure object
- `efficiency_data`: List of calculated efficiency data for each classifier
"""
function compare_efficiency_vs_momentum(
    scores_list::Vector{<:AbstractVector{<:Real}},
    labels_list::Vector{<:AbstractVector{<:Integer}},
    momentum_list::Vector{<:AbstractVector{<:Real}},
    thresholds::Vector{<:Real},
    bin_edges::AbstractVector{<:Real};
    labels = nothing,
    title = "Efficiency Comparison",
    xlabel = "Momentum",
    ylabel = "Efficiency",
    figsize = (600, 400),
    colors = nothing,
    min_samples = 10,
    legend_position = :lb,
    show_bin_edges = true,
    tick_format = x -> string(Int(round(x))),
)
    n_configs = length(scores_list)
    @assert length(labels_list) == n_configs "Must provide same number of label vectors as score vectors"
    @assert length(momentum_list) == n_configs "Must provide same number of momentum vectors as score vectors"
    @assert length(thresholds) == n_configs "Must provide one threshold per classifier"

    # Default labels if not provided
    if isnothing(labels)
        labels = ["Classifier $i" for i = 1:n_configs]
    end

    # Default colors if not provided
    if isnothing(colors)
        colormap = cgrad(:Dark2_8, n_configs, categorical = true)
        colors = [colormap[i] for i = 1:n_configs]
    end

    # Create figure
    fig = Figure(size = figsize)
    ax = Axis(
        fig[1, 1],
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        titlesize = 20,
        xlabelsize = 16,
        ylabelsize = 16,
        xticklabelsize = 14,
        yticklabelsize = 14,
    )

    # Calculate and plot efficiency for each configuration
    eff_data_list = []
    for i = 1:n_configs
        eff_data = efficiency_per_momentum_bin(
            scores_list[i],
            labels_list[i],
            momentum_list[i],
            thresholds[i],
            bin_edges,
        )
        push!(eff_data_list, eff_data)

        # Filter out bins with too few samples
        valid_bins = eff_data.n_pos .>= min_samples
        x_values = eff_data.bin_centers[valid_bins]
        y_values = eff_data.efficiency[valid_bins]
        errors = eff_data.efficiency_error[valid_bins]

        # Plot efficiency points with error bars
        errorbars!(ax, x_values, y_values, errors, color = colors[i], whiskerwidth = 10)
        scatter!(
            ax,
            x_values,
            y_values,
            color = colors[i],
            markersize = 8,
            label = labels[i],
        )

        # Connect points with lines where data exists
        valid_mask = .!isnan.(y_values)
        if any(valid_mask)
            lines!(
                ax,
                x_values[valid_mask],
                y_values[valid_mask],
                color = colors[i],
                linewidth = 2,
            )
        end
    end

    # Add grid lines
    ax.xgridvisible = true
    ax.ygridvisible = true

    # Set y-axis limits from 0 to 1 for efficiency
    ax.limits = (nothing, nothing, 0, 1.05)

    # Set y-axis ticks to 0, 0.2, 0.4, 0.6, 0.8, 1.0
    ax.yticks = 0:0.2:1.0

    # Set x-axis ticks based on bin edges if requested
    if show_bin_edges
        ax.xticks = (bin_edges, map(tick_format, bin_edges))
    end

    # Add legend directly to the figure rather than to a specific subplot position
    axislegend(
        ax,
        position = legend_position,
        fontsize = 20,
        framecolor = :black,
        framealpha = 0.1,
    )

    return (figure = fig, efficiency_data = eff_data_list)
end

"""
    compare_per_bin_misid_efficiencies(
        scores_list::Vector{<:AbstractVector{<:Real}},
        labels_list::Vector{<:AbstractVector{<:Integer}},
        momentum_list::Vector{<:AbstractVector{<:Real}},
        misid_rate::Real,
        bin_edges::AbstractVector{<:Real};
        labels = nothing,
        title = "Efficiency Comparison (Per-Bin MisID)",
        xlabel = "Momentum",
        ylabel = "Efficiency",
        figsize = (600, 400),
        colors = nothing,
        min_bin_samples = 10,
        show_bin_edges = true,
        tick_format = x -> string(Int(round(x))),
        legend_position = :rb
    )

Compare the efficiency vs momentum curves for multiple classifiers using
per-bin misID rate optimization. This calculates separate thresholds for each
momentum bin to achieve the target misID rate within that bin.

# Arguments
- `scores_list`: List of score vectors for each classifier
- `labels_list`: List of label vectors for each classifier
- `momentum_list`: List of momentum vectors for each classifier
- `misid_rate`: Target misidentification rate (e.g., 0.01 for 1%)
- `bin_edges`: Vector of bin edges for momentum bins
- `labels`: Optional list of labels for the legend
- `title`: Plot title
- `xlabel`: X-axis label
- `ylabel`: Y-axis label
- `figsize`: Figure size in pixels
- `colors`: Optional list of colors for each classifier
- `min_bin_samples`: Minimum number of samples required in a bin for calculation
- `show_bin_edges`: Whether to show bin edges as x-axis ticks
- `tick_format`: Function to format tick labels
- `legend_position`: Position of the legend

# Returns
A tuple containing:
- `figure`: The Figure object
- `bin_data_list`: List of calculated bin-by-bin efficiency and misID data for each classifier
"""
function compare_per_bin_misid_efficiencies(
    scores_list::Vector{<:AbstractVector{<:Real}},
    labels_list::Vector{<:AbstractVector{<:Integer}},
    momentum_list::Vector{<:AbstractVector{<:Real}},
    misid_rate::Real,
    bin_edges::AbstractVector{<:Real};
    labels = nothing,
    title = "Efficiency Comparison (Per-Bin MisID)",
    xlabel = "Momentum",
    ylabel = "Efficiency",
    figsize = (600, 400),
    colors = nothing,
    min_bin_samples = 10,
    show_bin_edges = true,
    tick_format = x -> string(Int(round(x))),
    legend_position = :rb,
)
    n_configs = length(scores_list)
    @assert length(labels_list) == n_configs "Must provide same number of label vectors as score vectors"
    @assert length(momentum_list) == n_configs "Must provide same number of momentum vectors as score vectors"

    # Default labels if not provided
    if isnothing(labels)
        labels = ["Classifier $i" for i = 1:n_configs]
    end

    # Default colors if not provided
    if isnothing(colors)
        colormap = cgrad(:Dark2_8, n_configs, categorical = true)
        colors = [colormap[i] for i = 1:n_configs]
    end

    # Update title to include misID rate
    title = "$title (Target MisID: $(100*misid_rate)%)"

    # Calculate per-bin efficiency for each classifier
    bin_data_list = []
    bin_centers_list = []
    efficiency_list = []
    efficiency_error_list = []

    for i = 1:n_configs
        bin_data = efficiency_per_momentum_bin_at_misid_rate(
            scores_list[i],
            labels_list[i],
            momentum_list[i],
            misid_rate,
            bin_edges;
            min_bin_samples = min_bin_samples,
        )

        push!(bin_data_list, bin_data)
        push!(bin_centers_list, bin_data.bin_centers)
        push!(efficiency_list, bin_data.efficiency)
        push!(efficiency_error_list, bin_data.efficiency_error)
    end

    # Create comparison plot
    result = compare_bin_efficiency_data(
        bin_centers_list,
        efficiency_list,
        efficiency_error_list,
        bin_edges;
        labels = labels,
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        figsize = figsize,
        colors = colors,
        show_bin_edges = show_bin_edges,
        tick_format = tick_format,
        legend_position = legend_position,
    )

    return (figure = result.figure, bin_data_list = bin_data_list)
end

"""
    compare_performance_curve(...)

Compares the performance curves of different models or methods.

# Arguments
- `...`: Arguments required for the comparison (please specify the actual arguments).

# Description
This function generates and compares performance curves, such as ROC or efficiency curves, for multiple models or datasets. It is useful for visualizing and evaluating the relative performance of different approaches.

# Returns
Returns a plot or data structure representing the comparison of performance curves.
"""
function compare_performance_curve(
    scores_list::Vector{<:AbstractVector{<:Real}},
    labels_list::Vector{<:AbstractVector{<:Integer}},
    legends_list::Vector{String},
    colors_list::Vector{Symbol};
    #bin_edges_x::Union{AbstractVector{<:Real},Nothing}=nothing,
    #bin_edges_y::Union{AbstractVector{<:Real},Nothing}=nothing,
    kwargs...,
)

    title = get(kwargs, :title, "Misid Probability vs Efficiency")
    xlabel = get(kwargs, :xlabel, "Efficiency")
    ylabel = get(kwargs, :ylabel, "Misid Probability")
    legend_position = get(kwargs, :legend_position, :lt)
    logy = get(kwargs, :logy, false)
    figsize = get(kwargs, :figsize, (600, 400))

    n_configs = length(scores_list)
    @assert length(labels_list) == n_configs "Must provide same number of label vectors as score vectors"

    fig = Figure(size = figsize)
    ax = Axis(
        fig[1, 1],
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        titlesize = 20,
        xlabelsize = 16,
        ylabelsize = 16,
        xticklabelsize = 14,
        yticklabelsize = 14,
    )

    fig_log = Figure(size = figsize)
    ax_log = Axis(
        fig_log[1, 1],
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        titlesize = 20,
        xlabelsize = 16,
        ylabelsize = 16,
        xticklabelsize = 14,
        yticklabelsize = 14,
        yscale = log10,
        yminorgridvisible = true,
    )

    for i = 1:n_configs
        misid_eff = misid_eff_dataframe(scores_list[i], labels_list[i]; compress = true)
        lines!(
            ax,
            misid_eff.efficiency,
            misid_eff.misid;
            label = legends_list[i],
            linewidth = 2,
            linestyle = :solid,
            color = colors_list[i],
        )
        lines!(
            ax_log,
            misid_eff.efficiency,
            misid_eff.misid;
            label = legends_list[i],
            linewidth = 2,
            linestyle = :solid,
            color = colors_list[i],
        )
    end

    # Add grid lines
    ax.xgridvisible = true
    ax.ygridvisible = true

    ax_log.xgridvisible = true
    ax_log.ygridvisible = true

    # Set y-axis limits from 0 to 1 for efficiency
    ax.limits = (0, 1.0, 0, 1.0)
    ax_log.limits = (0, 1.0, 1e-2, 1.0)

    # Set x-axis ticks to 0, 0.2, 0.4, 0.6, 0.8, 1.0
    ax.xticks = 0:0.2:1.0
    ax_log.xticks = 0:0.2:1.0

    # Set y-axis ticks
    ax.yticks = 0:0.2:1.0
    ax_log.yticks = [0.01, 0.05, 0.1, 0.5, 1.0]

    axislegend(
        ax,
        position = legend_position,
        fontsize = 20,
        framecolor = :black,
        framealpha = 0.1,
    )
    axislegend(
        ax_log,
        position = legend_position,
        fontsize = 20,
        framecolor = :black,
        framealpha = 0.1,
    )
    return fig, fig_log
end

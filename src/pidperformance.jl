"""
    find_thresholds_for_misid(scores, labels; misid_rates=[0.10,0.01,0.001])

Given `scores` and binary `labels` (0 = negative, 1 = positive), return a Dict
mapping each requested misid rate (fraction) to a NamedTuple with:
  (threshold, achieved_misid, efficiency, fp, n_neg, tp, n_pos)

Threshold is chosen so that the number of negatives with `score > threshold`
is <= floor(misid_rate * n_neg) (i.e. misid ≤ requested rate when possible).

Example:
    res = find_thresholds_for_misid(scores, labels)
    res[0.01].threshold
"""
function find_workingpoints(
    scores::AbstractVector{<:Real},
    labels::AbstractVector{<:Integer};
    misid_rates = [0.10, 0.01, 0.001],
)
    @assert length(scores) == length(labels)
    # Separate scores
    neg_scores = collect(scores[labels.==0])
    pos_scores = collect(scores[labels.==1])
    n_neg = length(neg_scores)
    n_pos = length(pos_scores)

    if n_neg == 0
        error("No negative samples found (labels==0). Cannot compute misid thresholds.")
    end
    if n_pos == 0
        error("No positive samples found (labels==1). Cannot compute efficiency.")
    end

    sort_neg = sort(neg_scores)  # ascending

    results = Dict{
        Float64,
        NamedTuple{
            (:threshold, :achieved_misid, :efficiency, :fp, :n_neg, :tp, :n_pos),
            Tuple{Float64,Float64,Float64,Int,Int,Int,Int},
        },
    }()

    for r in misid_rates
        if !(0.0 <= r <= 1.0)
            error("misid rate must be between 0 and 1; got $r")
        end

        # allowed false positives (negatives above threshold)
        fp_allowed = floor(Int, r * n_neg)
        idx = clamp(n_neg - fp_allowed, 1, n_neg)     # index into ascending sorted negatives
        threshold = Float64(sort_neg[idx])

        # compute achieved counts using threshold rule: predict positive if score > threshold
        fp = count(x -> x > threshold, neg_scores)
        tp = count(x -> x > threshold, pos_scores)

        achieved_misid = fp / n_neg
        efficiency = tp / n_pos

        results[r] = (
            threshold = threshold,
            achieved_misid = achieved_misid,
            efficiency = efficiency,
            fp = fp,
            n_neg = n_neg,
            tp = tp,
            n_pos = n_pos,
        )
    end

    return results
end

"""
    _pos_neg_counts(labels)

Return boolean masks and counts for positive (label==1) and negative (label==0) samples.
"""
function _pos_neg_counts(labels::AbstractVector{<:Integer})
    pos_mask = labels .== 1
    neg_mask = labels .== 0
    n_pos = count(pos_mask)
    n_neg = count(neg_mask)
    return pos_mask, neg_mask, n_pos, n_neg
end

"""
    counts_at_threshold(scores, labels, threshold; pred_gt=true)

Compute (tp, fp, tn, fn) using the decision rule `predict positive if score > threshold` by default.
If `pred_gt=false` uses `>=` instead.
"""
function counts_at_threshold(
    scores::AbstractVector{<:Real},
    labels::AbstractVector{<:Integer},
    threshold::Real;
    pred_gt::Bool = true,
)
    @assert length(scores) == length(labels)
    if pred_gt
        preds = scores .> threshold
    else
        preds = scores .>= threshold
    end
    tp = count(i -> (preds[i] && labels[i] == 1), 1:length(scores))
    fp = count(i -> (preds[i] && labels[i] == 0), 1:length(scores))
    tn = count(i -> (!preds[i] && labels[i] == 0), 1:length(scores))
    fn = count(i -> (!preds[i] && labels[i] == 1), 1:length(scores))
    return tp, fp, tn, fn
end

"""
    misid_eff_points(scores, labels; compress=true)

Compute arrays of (misid_probability, efficiency, threshold) for the given `scores` and binary `labels`
(0 = negative, 1 = positive). This is computed efficiently by sorting scores descending and using
cumulative sums. By default the result is compressed to unique threshold steps; set `compress=false`
to obtain a point per sample.

Returns a NamedTuple with fields `(misid, efficiency, threshold)` where each is a Vector.
"""
function misid_eff_points(
    scores::AbstractVector{<:Real},
    labels::AbstractVector{<:Integer};
    compress::Bool = true,
)
    @assert length(scores) == length(labels)
    _, _, n_pos, n_neg = _pos_neg_counts(labels)
    if n_pos == 0 || n_neg == 0
        error("labels must contain both positive (1) and negative (0) examples")
    end

    # sort scores descending and reorder labels accordingly
    inds = sortperm(scores; rev = true)
    sorted_scores = scores[inds]
    sorted_labels = labels[inds]

    # cumulative true positives at each cut (predict positive for top-k)
    cum_pos = cumsum(sorted_labels .== 1)
    # number of predictions at each cut = 1:length(scores)
    ks = collect(1:length(scores))
    cum_fp = ks .- cum_pos

    misid = cum_fp ./ n_neg
    efficiency = cum_pos ./ n_pos
    thresholds = sorted_scores

    if compress
        # keep only points where threshold changes (unique score steps)
        keep = vcat(trues(1), diff(thresholds) .!= 0)
        return (
            misid = misid[keep],
            efficiency = efficiency[keep],
            threshold = thresholds[keep],
        )
    else
        return (misid = misid, efficiency = efficiency, threshold = thresholds)
    end
end

function misid_eff_dataframe(scores, labels; compress::Bool = true)
    pts = misid_eff_points(scores, labels; compress = compress)
    # Try to build a DataFrame if available
    try
        @eval using DataFrames
        return DataFrame(
            misid = pts.misid,
            efficiency = pts.efficiency,
            threshold = pts.threshold,
        )
    catch
        return pts
    end
end

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


"""
    efficiency_per_momentum_bin(scores, labels, momentum, threshold, bin_edges)

Calculate the efficiency in each momentum bin for a given classification threshold.

# Arguments
- `scores`: Vector of classifier scores
- `labels`: Vector of true labels (0 = negative, 1 = positive)
- `momentum`: Vector of momentum values for each sample
- `threshold`: Classification threshold (predict positive if score > threshold)
- `bin_edges`: Vector of bin edges for momentum bins

# Returns
A NamedTuple with fields:
- `bin_centers`: Centers of momentum bins
- `efficiency`: Efficiency in each bin
- `efficiency_error`: Binomial error on the efficiency
- `n_pos`: Number of positive samples in each bin
- `n_tot`: Total number of samples in each bin
"""
function efficiency_per_momentum_bin(
    scores::AbstractVector{<:Real},
    labels::AbstractVector{<:Integer},
    momentum::AbstractVector{<:Real},
    threshold::Real,
    bin_edges::AbstractVector{<:Real},
)
    @assert length(scores) == length(labels) == length(momentum) "All input vectors must have the same length"

    n_bins = length(bin_edges) - 1
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i = 1:n_bins]

    # Initialize arrays to store results
    efficiencies = zeros(n_bins)
    efficiency_errors = zeros(n_bins)
    n_pos_bins = zeros(Int, n_bins)
    n_tot_bins = zeros(Int, n_bins)

    # Predictions based on threshold
    predictions = scores .> threshold

    # Find bin for each sample
    bin_indices = zeros(Int, length(momentum))
    for i = 1:length(momentum)
        # Find bin index (clamp to valid range)
        bin_idx = searchsortedfirst(bin_edges, momentum[i]) - 1
        bin_idx = clamp(bin_idx, 1, n_bins)
        bin_indices[i] = bin_idx
    end

    # Calculate efficiency in each bin
    for bin = 1:n_bins
        # Get samples in this bin
        bin_mask = bin_indices .== bin
        bin_labels = labels[bin_mask]
        bin_preds = predictions[bin_mask]

        # Count positives and total samples in this bin
        pos_mask = bin_labels .== 1
        n_pos = sum(pos_mask)
        n_tot = length(bin_labels)

        # Store counts
        n_pos_bins[bin] = n_pos
        n_tot_bins[bin] = n_tot

        # Calculate efficiency and its error
        if n_pos > 0
            # True positives
            tp = sum(bin_preds[pos_mask])
            efficiency = tp / n_pos

            # Binomial error for efficiency
            # σ = √(p(1-p)/n) where p is the efficiency and n is the number of positive samples
            error = sqrt((efficiency * (1 - efficiency)) / n_pos)

            efficiencies[bin] = efficiency
            efficiency_errors[bin] = error
        else
            # No positive samples in this bin
            efficiencies[bin] = NaN
            efficiency_errors[bin] = NaN
        end
    end

    return (
        bin_centers = bin_centers,
        efficiency = efficiencies,
        efficiency_error = efficiency_errors,
        n_pos = n_pos_bins,
        n_tot = n_tot_bins,
    )
end

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
    ax.yticks = 0:0.2:1.0

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
    legend_position = :rb,
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
    axislegend(ax, position = :cc, fontsize = 20, framecolor = :black, framealpha = 0.1)

    return (figure = fig, efficiency_data = eff_data_list)
end

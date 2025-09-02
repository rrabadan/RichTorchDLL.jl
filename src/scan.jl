using CairoMakie: Figure, Axis, heatmap!, Colorbar, scatter!, text!, Label
using Statistics: mean, std

"""
    parameter_scan(rich, torch, labels; w_range=-2.0:0.1:2.0, b_range=-2.0:0.1:2.0, verbose=true)

Performs a parameter scan over specified ranges for weights (`w_range`) and biases (`b_range`) using the provided `rich` and `torch` dlls and a set of `labels`.

# Arguments
- `rich`: The RICH DLLs.
- `torch`: The TORCH DLLs.
- `labels`: Labels or identifiers for the scan (0, 1).

# Keyword Arguments
- `w_range`: Range of weight values to scan over (default: -2.0:0.1:2.0).
- `b_range`: Range of bias values to scan over (default: -2.0:0.1:2.0).
- `verbose`: If `true`, prints progress and additional information during the scan (default: `true`).

# Returns
Returns the results of the parameter scan, as a data structure containing performance metrics or scan outcomes for each parameter combination.
"""
function parameter_scan(
    rich,
    torch,
    labels;
    w_range = -2.0:0.1:2.0,
    b_range = -2.0:0.1:2.0,
    verbose = true,
)
    # Create containers for results
    n_w = length(w_range)
    n_b = length(b_range)

    # Metrics to track
    efficiencies = zeros(n_w, n_b)  # Efficiency (TP/P)
    purities = zeros(n_w, n_b)      # Purity (TP/(TP+FP))
    misids = zeros(n_w, n_b)        # Misidentification probability (FP/N)
    f1_scores = zeros(n_w, n_b)     # F1 score
    aucs = zeros(n_w, n_b)          # AUC

    # Indices of positive and negative samples
    pos_indices = findall(labels .== 1)
    neg_indices = findall(labels .== 0)
    n_pos = length(pos_indices)
    n_neg = length(neg_indices)

    if verbose
        println("Scanning $(n_w*n_b) parameter combinations...")
        println("Positive samples: $n_pos, Negative samples: $n_neg")
    end

    # Perform grid search
    Threads.@threads for i = 1:n_w
        for j = 1:n_b
            w = w_range[i]
            b = b_range[j]

            # Apply model: rich + (w * torch + b)
            scores = rich .+ (w .* torch .+ b)

            # Calculate metrics using threshold at 0
            predictions = scores .> 0.01

            # True positives
            tp = sum(predictions[pos_indices])
            # False positives
            fp = sum(predictions[neg_indices])

            # Calculate metrics
            efficiency = tp / n_pos
            purity = tp > 0 ? tp / (tp + fp) : 0.0
            misid = fp / n_neg  # Misidentification probability

            # F1 score (harmonic mean of efficiency and purity)
            f1 =
                (efficiency > 0 && purity > 0) ?
                2 * (efficiency * purity) / (efficiency + purity) : 0.0

            # Also calculate AUC (stratified sampling)
            auc = calculate_auc_stratified_sampled(scores, labels, max_pairs = 10000)

            # Store results
            efficiencies[i, j] = efficiency
            purities[i, j] = purity
            misids[i, j] = misid
            f1_scores[i, j] = f1
            aucs[i, j] = auc
        end
    end

    # Find optimal parameters for efficiency
    max_eff_idx = argmax(efficiencies)
    max_eff_w = w_range[max_eff_idx[1]]
    max_eff_b = b_range[max_eff_idx[2]]
    max_eff = efficiencies[max_eff_idx]

    # Find minimum misid rate (best parameters for lowest misid)
    min_misid_idx = argmin(misids)
    min_misid_w = w_range[min_misid_idx[1]]
    min_misid_b = b_range[min_misid_idx[2]]
    min_misid = misids[min_misid_idx]

    # Find optimal parameters for F1 score (balance of efficiency and purity)
    max_f1_idx = argmax(f1_scores)
    max_f1_w = w_range[max_f1_idx[1]]
    max_f1_b = b_range[max_f1_idx[2]]
    max_f1 = f1_scores[max_f1_idx]

    # Find optimal parameters for AUC
    max_auc_idx = argmax(aucs)
    max_auc_w = w_range[max_auc_idx[1]]
    max_auc_b = b_range[max_auc_idx[2]]
    max_auc = aucs[max_auc_idx]

    if verbose
        println("\nResults:")
        println("Max Efficiency: $max_eff at w=$max_eff_w, b=$max_eff_b")
        println("Min Misid: $min_misid at w=$min_misid_w, b=$min_misid_b")
        println("Max F1 Score: $max_f1 at w=$max_f1_w, b=$max_f1_b")
        println("Max AUC: $max_auc at w=$max_auc_w, b=$max_auc_b")
    end

    # Return results and visualization data
    return (
        # Parameter grids
        w_range = w_range,
        b_range = b_range,

        # Result matrices
        efficiencies = efficiencies,
        purities = purities,
        misids = misids,
        f1_scores = f1_scores,
        aucs = aucs,

        # Best parameters
        best_efficiency = (w = max_eff_w, b = max_eff_b, value = max_eff),
        best_misid = (w = min_misid_w, b = min_misid_b, value = min_misid),
        best_f1 = (w = max_f1_w, b = max_f1_b, value = max_f1),
        best_auc = (w = max_auc_w, b = max_auc_b, value = max_auc),
    )
end

# Function to visualize scan results
"""
    visualize_scan_results!(fig, scan_results; metric=:auc)

Visualizes the results of a parameter scan using CairoMakie.

# Arguments
- `fig`: The Figure object to visualize the results on.
- `scan_results`: The results from running `parameter_scan` function.
- `metric`: Symbol specifying which metric to visualize (default is `:auc`).

# Returns
- A Figure object that can be further customized or saved.

# Description
Generates a heatmap visualization to help analyze the performance of different parameter configurations based on the specified metric.
"""
function visualize_scan_results!(fig, scan_results; metric = :auc)

    w_range = scan_results.w_range
    b_range = scan_results.b_range

    # Select metric
    if metric == :efficiency
        data = scan_results.efficiencies
        title_text = "Efficiency (TP/P)"
        best = scan_results.best_efficiency
        colormap = :viridis
    elseif metric == :misid
        data = scan_results.misids
        title_text = "Misidentification Probability (FP/N)"
        best = scan_results.best_misid
        colormap = :plasma
    elseif metric == :purity
        data = scan_results.purities
        title_text = "Purity (TP/(TP+FP))"
        best = scan_results.best_f1
        colormap = :turbo
    elseif metric == :f1
        data = scan_results.f1_scores
        title_text = "F1 Score"
        best = scan_results.best_f1
        colormap = :cividis
    else  # Default to AUC
        data = scan_results.aucs
        title_text = "AUC"
        best = scan_results.best_auc
        colormap = :viridis
    end

    # Create axis
    ax = Axis(
        fig,
        title = title_text,
        xlabel = "Weight (w)",
        ylabel = "Bias (b)",
        titlesize = 20,
        xlabelsize = 16,
        ylabelsize = 16,
        xticklabelsize = 14,
        yticklabelsize = 14,
    )

    # Create heatmap - transpose data to match orientation
    hm = heatmap!(ax, w_range, b_range, data', colormap = colormap)

    # Add colorbar
    Colorbar(fig[1, 2], hm, label = title_text)

    # Mark the best point
    scatter!(ax, [best.w], [best.b], color = :red, markersize = 15)

    # Add text annotation for best point
    text!(
        ax,
        "Best: ($(round(best.w, digits=2)), $(round(best.b, digits=2)))",
        position = (best.w, best.b),
        align = (:center, :bottom),
        offset = (0, 15),
        fontsize = 12,
    )

    # Add grid lines for better readability
    ax.xgridvisible = true
    ax.ygridvisible = true

    return ax
end

function visualize_scan_results(scan_results; metric = :auc, figsize = (800, 600))
    fig = Figure(size = figsize)
    visualize_scan_results!(fig[1, 1], scan_results; metric = metric)
    return fig
end

"""
    optimize_combination_model(rich, torch, labels; figsize=(1000, 800))

Optimize a combination model using the provided `rich` and `torch` data, along with the corresponding `labels`.

# Arguments
- `rich`: The RICH DLLs.
- `torch`: The TORCH DLLs.
- `labels`: Labels or identifiers for the scan (0, 1).
- `figsize`: Size of the figure in pixels (default is (1000, 800)).

# Returns
- A tuple containing (figure, scan_results) where figure is the visualization and scan_results are the numeric results.

# Description
This function performs a parameter scan to optimize the combination of RICH and TORCH DLLs,
and generates a multi-panel visualization of different performance metrics.
"""
function optimize_combination_model(rich, torch, labels; figsize = (1000, 800))

    # Parameter scan
    scan_results =
        parameter_scan(rich, torch, labels, w_range = -2.0:0.1:2.0, b_range = -2.0:0.1:2.0)

    # Create a 2x2 grid figure
    fig = Figure(size = figsize)

    # Define the metrics to visualize
    metrics = [:efficiency, :misid, :f1, :auc]
    titles = ["Efficiency", "Misidentification Rate", "F1 Score", "AUC"]

    # Generate the four panels
    for (i, metric) in enumerate(metrics)
        row = div(i - 1, 2) + 1
        col = mod(i - 1, 2) + 1

        # Select metric data
        if metric == :efficiency
            data = scan_results.efficiencies
            best = scan_results.best_efficiency
            colormap = :viridis
        elseif metric == :misid
            data = scan_results.misids
            best = scan_results.best_misid
            colormap = :plasma
        elseif metric == :f1
            data = scan_results.f1_scores
            best = scan_results.best_f1
            colormap = :turbo
        else  # Default to AUC
            data = scan_results.aucs
            best = scan_results.best_auc
            colormap = :viridis
        end

        # Create axis
        ax = Axis(
            fig[row, col],
            title = titles[i],
            xlabel = "Weight (w)",
            ylabel = "Bias (b)",
            titlesize = 16,
            xlabelsize = 14,
            ylabelsize = 14,
        )

        # Create heatmap
        hm = heatmap!(
            ax,
            scan_results.w_range,
            scan_results.b_range,
            data',
            colormap = colormap,
        )

        # Add colorbar
        Colorbar(fig[row, col+2], hm, label = titles[i])

        # Mark the best point
        scatter!(ax, [best.w], [best.b], color = :red, markersize = 10)

        # Add grid lines
        ax.xgridvisible = true
        ax.ygridvisible = true
    end

    # Add a common title
    Label(fig[0, 1:2], "Parameter Scan Results", fontsize = 20)

    # Return both the figure and the scan results
    return (figure = fig, results = scan_results)
end

# -- One-dimensional scan (fix one parameter, scan the other) -------------------
"""
    parameter_scan_1d(
        rich,
        torch,
        labels;
        fixed_value::Real=1.0,
        scan::Symbol=:w,
        scan_range=-2.0:0.1:2.0,
        threshold::Real=0.0,
        verbose::Bool=true,
    )

Performs a one-dimensional parameter scan over a specified parameter of the system.

# Arguments
- `rich`: The RICH DLL.
- `torch`: The TORCH DLL.
- `labels`: Labels or identifiers for the scan (0, 1)

# Keyword Arguments
- `fixed_value::Real=1.0`: The value to fix the non-scanned parameter(s) at during the scan.
- `scan::Symbol=:w`: The symbol of the parameter to scan (e.g., `:w` for width).
- `scan_range=-2.0:0.1:2.0`: The range of values to scan over for the selected parameter.
- `threshold::Real=0.0`: Threshold value for filtering or decision-making during the scan.
- `verbose::Bool=true`: If `true`, prints progress and additional information during the scan.

# Returns
Returns the results of the parameter scan, as a data structure containing performance metrics or scan outcomes for each parameter combination.
"""
function parameter_scan_1d(
    rich,
    torch,
    labels;
    scan::Symbol = :w,
    scan_range = -2.0:0.1:2.0,
    fixed_value::Real = 1.0,
    max_pairs_auc::Int = 10000,
    repeats_auc::Int = 1,
    threshold::Real = 0.0,
    verbose::Bool = true,
)
    @assert scan == :w || scan == :b

    n = length(scan_range)
    efficiency = zeros(n)
    purity = zeros(n)
    misid = zeros(n)
    f1 = zeros(n)
    auc = zeros(n)
    auc_std = zeros(n)

    pos_indices = findall(labels .== 1)
    neg_indices = findall(labels .== 0)
    n_pos = length(pos_indices)
    n_neg = length(neg_indices)

    if verbose
        println("1D scan over $(n) values of $(scan) (fixed other = $(fixed_value)))")
        println("AUC repeats: $repeats_auc")
    end

    for (i, param) in enumerate(scan_range)
        if scan == :w
            w = param
            b = fixed_value
        else
            w = fixed_value
            b = param
        end

        scores = rich .+ (w .* torch .+ b)
        preds = scores .> threshold

        tp = sum(preds[pos_indices])
        fp = sum(preds[neg_indices])

        efficiency[i] = tp / (n_pos > 0 ? n_pos : 1)
        purity[i] = (tp + fp) > 0 ? tp / (tp + fp) : 0.0
        misid[i] = fp / (n_neg > 0 ? n_neg : 1)
        f1[i] =
            (efficiency[i] > 0 && purity[i] > 0) ?
            2 * (efficiency[i] * purity[i]) / (efficiency[i] + purity[i]) : 0.0
        auc_results = calculate_auc_stratified_sampled(
            scores,
            labels,
            max_pairs = max_pairs_auc,
            repeats = repeats_auc,
        )
        if repeats_auc > 1
            auc[i], auc_std[i] = auc_results
        else
            auc[i] = auc_results
            auc_std[i] = 0.0
        end
    end

    # best by AUC
    best_idx = argmax(auc)
    best = (
        index = best_idx,
        param = scan_range[best_idx],
        auc = auc[best_idx],
        auc_std = auc_std[best_idx],
    )

    # minimum misid index
    min_misid_idx = argmin(misid)
    min_misid_val = misid[min_misid_idx]

    # For efficiency at fixed misid (e.g., find highest efficiency with misid <= 0.05)
    target_misid = 0.05  # 5% misid rate as example
    valid_indices = findall(misid .<= target_misid)

    # If there are any indices with misid <= target, find max efficiency among them
    best_eff_at_target = if !isempty(valid_indices)
        max_eff_idx = argmax(efficiency[valid_indices])
        param_idx = valid_indices[max_eff_idx]
        (
            index = param_idx,
            param = scan_range[param_idx],
            efficiency = efficiency[param_idx],
            misid = misid[param_idx],
        )
    else
        # If no valid points found, return nothing
        nothing
    end

    return (
        param_name = scan,
        param_range = scan_range,
        best = best,
        min_misid = (
            index = min_misid_idx,
            param = scan_range[min_misid_idx],
            misid = min_misid_val,
        ),
        best_eff_at_target_misid = best_eff_at_target,
        auc = auc,
        efficiency = efficiency,
        purity = purity,
        misid = misid,
        f1 = f1,
    )
end

"""
    visualize_1d_scan!(fig, scan1d; metric::Symbol = :auc, xlabel = nothing)

Visualizes the results of a 1D scan using CairoMakie.

# Arguments
- `fig`: The figure object to which the plot will be added.
- `scan1d`: The data structure containing the results of the 1D scan to visualize.
- `metric::Symbol`: (Optional) The metric to plot from the scan results. Default is `:auc`.
- `xlabel`: (Optional) Label for the x-axis. If not provided, a default label will be used.
- `legend_position`: (Optional) Position of the legend on the plot. Default is `:rt` (right top).

# Returns
- An Axis object containing the plot.

# Description
This function generates a plot to visualize the specified metric from a 1D scan result. It is useful for analyzing how the chosen metric varies with the scanned parameter.
"""
function visualize_1d_scan!(
    fig,
    scan1d;
    metric::Symbol = :auc,
    xlabel = nothing,
    limits = (nothing, nothing),
    legend_position = :rt,
)

    param_range = scan1d.param_range
    if metric == :efficiency
        y = scan1d.efficiency
        title_text = "Efficiency"
    elseif metric == :purity
        y = scan1d.purity
        title_text = "Purity"
    elseif metric == :misid
        y = scan1d.misid
        title_text = "Misidentification Rate"
    elseif metric == :f1
        y = scan1d.f1
        title_text = "F1 score"
    else
        y = scan1d.auc
        title_text = "AUC"
    end

    # Get best point for highlighting
    best = scan1d.best
    v = best.param
    best_index = best.index
    bestval = y[best_index]

    # Create axis
    ax = Axis(
        fig,
        limits = limits,
        title = "$(title_text) vs $(scan1d.param_name)",
        xlabel = xlabel === nothing ? string(scan1d.param_name) : xlabel,
        ylabel = title_text,
        titlesize = 20,
        xlabelsize = 16,
        ylabelsize = 16,
        xticklabelsize = 14,
        yticklabelsize = 14,
    )

    # Plot the main line
    lines!(ax, param_range, y, linewidth = 2, color = :royalblue)
    scatter!(ax, param_range, y, markersize = 5, color = :royalblue, label = title_text)

    # Highlight the best point
    scatter!(ax, [v], [bestval], color = :red, markersize = 10, label = "Best AUC")

    # Add a text annotation for the best point
    #text!(
    #    ax,
    #    "Best: ($(round(v, digits=2)), $(round(bestval, digits=3)))",
    #    position=(v, bestval),
    #    align=(:center, :bottom),
    #    offset=(0, 10),
    #    fontsize=12,
    #)

    # Add grid lines for better readability
    ax.xgridvisible = true
    ax.ygridvisible = true

    # Add legend
    axislegend(ax, position = legend_position)

    return ax
end

function visualize_1d_scan(
    scan1d;
    metric::Symbol = :auc,
    xlabel = nothing,
    limits = (nothing, nothing),
    legend_position = :rt,
    figsize = (800, 600),
)
    fig = Figure(size = figsize)
    visualize_1d_scan!(
        fig,
        scan1d;
        metric = metric,
        xlabel = xlabel,
        limits = limits,
        legend_position = legend_position,
    )
    return fig
end

"""
    repeated_parameter_scan(rich, torch, labels, scan_var, scan_range, fixed_value, output_dir, n_repeats; kwargs...)

Repeat the parameter scan n times and compute statistics on the optimal weights.

# Arguments
- `rich`: Vector of RICH DLL values
- `torch`: Vector of TORCH DLL values
- `labels`: Vector of binary labels (1 for signal, 0 for background)
- `scan_var`: The variable to scan (e.g., :w for weight, :b for bias)
- `scan_range`: The range of values to scan over
- `fixed_value`: The fixed value for the other parameter (e.g., bias when scanning weight)
- `n_repeats`: Number of scan repetitions
- `kwargs`: Additional keyword arguments passed to parameter_scan_1d

# Returns
A named tuple containing:
- `mean`: Mean of optimal weight values across all scans
- `std`: Standard deviation of optimal weight values
- `bestvalues`: Vector of all optimal weight values
"""
function repeated_parameter_scan(
    rich::Vector{<:Real},
    torch::Vector{<:Real},
    labels::Vector{Int},
    scan_var::Symbol,
    scan_range::StepRangeLen{Float64,Base.TwicePrecision{Float64}},
    fixed_value::Float64,
    n_repeats::Int,
    savefig_func::Function;
    kwargs...,
)

    figsize = get(kwargs, :figsize, (1000, 400))

    scan_dir = "scans_$(scan_var)"
    best_values = Float64[]

    for i = 1:n_repeats
        println("Running parameter scan $i of $n_repeats...")

        # Run the scan
        scan_result = parameter_scan_1d(
            rich,
            torch,
            labels;
            scan = scan_var,
            scan_range = scan_range,
            fixed_value = fixed_value,
            verbose = false,
            kwargs...,
        )

        # Save the best weight
        push!(best_values, scan_result.best.param)

        # Create visualization of this scan
        fig = Figure(size = figsize)
        ax_auc = visualize_1d_scan!(
            fig[1, 1],
            scan_result,
            limits = (nothing, (0.0, 1.1)),
            legend_position = :lt,
        )
        ax_misid = visualize_1d_scan!(
            fig[1, 2],
            scan_result,
            metric = :misid,
            limits = (nothing, (0.0, 1.0)),
            legend_position = :rt,
        )
        ax_eff = visualize_1d_scan!(
            fig[1, 3],
            scan_result,
            metric = :efficiency,
            limits = (nothing, (0.0, 1.0)),
            legend_position = :rb,
        )

        # Save the figure for this scan
        savefig_func(fig, "$(scan_dir)/scan_$(i)")
    end

    # Calculate statistics
    mean_ = mean(best_values)
    std_ = std(best_values)

    # Create a histogram of the best parameters
    fig1 = Figure(size = (800, 600))
    ax = Axis(
        fig1[1, 1],
        title = "Distribution of Optimal Parameters (n=$n_repeats)",
        xlabel = (scan_var == :w ? "Weight (w)" : "Bias (b)"),
        ylabel = "",
    )

    hist!(ax, best_values, bins = min(20, n_repeats))

    # Add vertical line for mean
    vlines!(
        ax,
        [mean_],
        color = :red,
        linewidth = 2,
        label = "Mean = $(round(mean_, digits=3)) ± $(round(std_, digits=3))",
    )

    axislegend(position = :lb)

    # Save the histogram
    savefig_func(fig1, "$(scan_dir)/$(scan_var)_distribution")

    println("\nRepeated Parameter Scan Results:")
    println("Mean weight: $(round(mean_, digits=4))")
    println("Std dev: $(round(std_, digits=4))")
    println(
        "Range: [$(round(minimum(best_values), digits=4)), $(round(maximum(best_values), digits=4))]",
    )

    return (mean = mean_, std = std_, bestvalues = best_values)
end

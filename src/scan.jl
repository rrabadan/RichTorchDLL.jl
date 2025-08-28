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
    visualize_scan_results(scan_results; metric=:auc)

Visualizes the results of a parameter scan.

# Arguments
- `scan_results`: The results from running `parameter_scan` function.
- `metric`: Symbol specifying which metric to visualize (default is `:auc`).

# Description
Generates a plot or visualization to help analyze the performance of different parameter configurations based on the specified metric.
"""
function visualize_scan_results(scan_results; metric = :auc)
    w_range = scan_results.w_range
    b_range = scan_results.b_range

    # Select metric
    if metric == :efficiency
        data = scan_results.efficiencies
        title = "Efficiency (TP/P)"
        best = scan_results.best_efficiency
    elseif metric == :misid
        data = scan_results.misids
        title = "Misidentification Probability (FP/N)"
        best = scan_results.best_misid
    elseif metric == :purity
        data = scan_results.purities
        title = "Purity (TP/(TP+FP))"
        best = scan_results.best_f1
    elseif metric == :f1
        data = scan_results.f1_scores
        title = "F1 Score"
        best = scan_results.best_f1
    else  # Default to AUC
        data = scan_results.aucs
        title = "AUC"
        best = scan_results.best_auc
    end

    # Create heatmap
    heatmap(
        w_range,
        b_range,
        data',
        xlabel = "w",
        ylabel = "b",
        title = title,
        color = :viridis,
    )

    # Mark the best point
    scatter!(
        [best.w],
        [best.b],
        color = :red,
        markersize = 5,
        label = "Best: ($(best.w), $(best.b))",
    )

    # Return the plot for further customization
    current()
end

"""
    optimize_combination_model(rich, torch, labels)

Optimize a combination model using the provided `rich` and `torch` data, along with the corresponding `labels`.

# Arguments
- `rich`: The RICH DLLs.
- `torch`: The TORCH DLLs.
- `labels`: Labels or identifiers for the scan (0, 1).

# Returns
Returns the results of the scan over (w, b) parameter space.
"""
function optimize_combination_model(rich, torch, labels)
    # Parameter scan
    scan_results =
        parameter_scan(rich, torch, labels, w_range = -2.0:0.1:2.0, b_range = -2.0:0.1:2.0)

    # Visualize efficiency
    p1 = visualize_scan_results(scan_results, metric = :efficiency)

    # Visualize misidentification probability
    p2 = visualize_scan_results(scan_results, metric = :misid)

    # Visualize F1 score (balance of efficiency and purity)
    p3 = visualize_scan_results(scan_results, metric = :f1)

    # Visualize AUC
    p4 = visualize_scan_results(scan_results, metric = :auc)

    # Combine plots
    plot(p1, p2, p3, p4, layout = (2, 2), size = (1000, 800))

    # Return optimal parameters
    return scan_results
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
    fixed_value::Real = 1.0,
    scan::Symbol = :w,
    scan_range = -2.0:0.1:2.0,
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

    pos_indices = findall(labels .== 1)
    neg_indices = findall(labels .== 0)
    n_pos = length(pos_indices)
    n_neg = length(neg_indices)

    if verbose
        println("1D scan over $(n) values of $(scan) (fixed other = $(fixed_value)))")
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
        auc[i] = calculate_auc_stratified_sampled(scores, labels, max_pairs = 10000)
    end

    # best by AUC
    best_idx = argmax(auc)
    best = (index = best_idx, param = scan_range[best_idx], auc = auc[best_idx])

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
    visualize_1d_scan(scan1d; metric::Symbol = :auc, xlabel = nothing)

Visualizes the results of a 1D scan.

# Arguments
- `scan1d`: The data structure containing the results of the 1D scan to visualize.
- `metric::Symbol`: (Optional) The metric to plot from the scan results. Default is `:auc`.
- `xlabel`: (Optional) Label for the x-axis. If not provided, a default label will be used.

# Description
This function generates a plot to visualize the specified metric from a 1D scan result. It is useful for analyzing how the chosen metric varies with the scanned parameter.
"""
function visualize_1d_scan(scan1d; metric::Symbol = :auc, xlabel = nothing)
    param_range = scan1d.param_range
    if metric == :efficiency
        y = scan1d.efficiency
        title = "Efficiency"
    elseif metric == :purity
        y = scan1d.purity
        title = "Purity"
    elseif metric == :misid
        y = scan1d.misid
        title = "Misidentification Rate"
    elseif metric == :f1
        y = scan1d.f1
        title = "F1 score"
    else
        y = scan1d.auc
        title = "AUC"
    end

    p = plot(param_range, y, lw = 2, marker = :circle, label = string(title))
    best = scan1d.best
    v = best.param
    best_index = best.index
    bestval = y[best_index]
    scatter!([v], [bestval], color = :red, markersize = 6, label = "best AUC")

    if xlabel === nothing
        xlabel = string(scan1d.param_name)
    end
    xlabel!(xlabel)
    ylabel!(title)
    title!("$(title) vs $(scan1d.param_name)")

    return p
end

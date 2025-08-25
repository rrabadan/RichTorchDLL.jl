function parameter_scan(
    rich,
    torch,
    labels;
    w_range=-2.0:0.1:2.0,
    b_range=-2.0:0.1:2.0,
    verbose=true,
)
    # Create containers for results
    n_w = length(w_range)
    n_b = length(b_range)

    # Metrics to track
    efficiencies = zeros(n_w, n_b)  # Efficiency (TP/P)
    purities = zeros(n_w, n_b)      # Purity (TP/(TP+FP))
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

            # F1 score (harmonic mean of efficiency and purity)
            f1 =
                (efficiency > 0 && purity > 0) ?
                2 * (efficiency * purity) / (efficiency + purity) : 0.0

            # Also calculate AUC for reference
            auc = calculate_auc_sampled(scores, labels, max_pairs=10000)

            # Store results
            efficiencies[i, j] = efficiency
            purities[i, j] = purity
            f1_scores[i, j] = f1
            aucs[i, j] = auc
        end
    end

    # Find optimal parameters for efficiency
    max_eff_idx = argmax(efficiencies)
    max_eff_w = w_range[max_eff_idx[1]]
    max_eff_b = b_range[max_eff_idx[2]]
    max_eff = efficiencies[max_eff_idx]

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
        println("Max F1 Score: $max_f1 at w=$max_f1_w, b=$max_f1_b")
        println("Max AUC: $max_auc at w=$max_auc_w, b=$max_auc_b")
    end

    # Return results and visualization data
    return (
        # Parameter grids
        w_range=w_range,
        b_range=b_range,

        # Result matrices
        efficiencies=efficiencies,
        purities=purities,
        f1_scores=f1_scores,
        aucs=aucs,

        # Best parameters
        best_efficiency=(w=max_eff_w, b=max_eff_b, value=max_eff),
        best_f1=(w=max_f1_w, b=max_f1_b, value=max_f1),
        best_auc=(w=max_auc_w, b=max_auc_b, value=max_auc),
    )
end

# Function to visualize scan results
function visualize_scan_results(scan_results; metric=:auc)
    w_range = scan_results.w_range
    b_range = scan_results.b_range

    # Select metric
    if metric == :efficiency
        data = scan_results.efficiencies
        title = "Efficiency (TP/P)"
        best = scan_results.best_efficiency
    elseif metric == :purity
        data = scan_results.purities
        title = "Purity (TP/(TP+FP))"
        best = scan_results.best_f1  # Use F1 best point as proxy
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
        xlabel="w",
        ylabel="b",
        title=title,
        color=:viridis,
    )

    # Mark the best point
    scatter!(
        [best.w],
        [best.b],
        color=:red,
        markersize=5,
        label="Best: ($(best.w), $(best.b))",
    )

    # Return the plot for further customization
    current()
end

function optimize_combination_model(rich, torch, labels)
    # Parameter scan
    scan_results =
        parameter_scan(rich, torch, labels, w_range=-2.0:0.1:2.0, b_range=-2.0:0.1:2.0)

    # Visualize efficiency
    p1 = visualize_scan_results(scan_results, metric=:efficiency)

    # Visualize F1 score (balance of efficiency and purity)
    p2 = visualize_scan_results(scan_results, metric=:f1)

    # Visualize AUC
    p3 = visualize_scan_results(scan_results, metric=:auc)

    # Combine plots
    plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))

    # Return optimal parameters
    return scan_results
end

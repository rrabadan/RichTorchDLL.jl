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
    efficiency_per_momentum_bin_at_misid_rate(
        scores::AbstractVector{<:Real},
        labels::AbstractVector{<:Integer},
        momentum::AbstractVector{<:Real},
        misid_rate::Real,
        bin_edges::AbstractVector{<:Real};
        min_bin_samples::Int = 10
    )

Calculate efficiency for each momentum bin at a specific misID rate.
Unlike the global approach that uses a single threshold, this function
calculates a separate threshold for each momentum bin to achieve the target
misID rate within that bin.

# Arguments
- `scores`: Vector of classifier scores
- `labels`: Vector of true labels (0 = negative, 1 = positive)
- `momentum`: Vector of momentum values
- `misid_rate`: Target misidentification rate (e.g., 0.01 for 1%)
- `bin_edges`: Vector of bin edges for momentum bins
- `min_bin_samples`: Minimum number of samples (pos or neg) required in a bin for valid calculation

# Returns
A named tuple containing:
- `bin_centers`: Centers of momentum bins
- `efficiency`: Efficiency in each bin at the target misID rate
- `efficiency_error`: Error on the efficiency
- `thresholds`: Threshold used in each bin to achieve the target misID rate
- `achieved_misid`: Actual misID rate achieved in each bin
- `misid_error`: Error on the misID rate
- `n_pos`: Number of positive samples in each bin
- `n_neg`: Number of negative samples in each bin
"""
function efficiency_per_momentum_bin_at_misid_rate(
    scores::AbstractVector{<:Real},
    labels::AbstractVector{<:Integer},
    momentum::AbstractVector{<:Real},
    misid_rate::Real,
    bin_edges::AbstractVector{<:Real};
    min_bin_samples::Int = 10,
)
    @assert length(scores) == length(labels) == length(momentum) "All input vectors must have the same length"

    n_bins = length(bin_edges) - 1
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i = 1:n_bins]

    # Initialize arrays to store results
    efficiencies = zeros(n_bins)
    efficiency_errors = zeros(n_bins)
    thresholds = zeros(n_bins)
    achieved_misids = zeros(n_bins)
    misid_errors = zeros(n_bins)
    n_pos_bins = zeros(Int, n_bins)
    n_neg_bins = zeros(Int, n_bins)

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
        bin_scores = scores[bin_mask]
        bin_labels = labels[bin_mask]

        # Count positives and negatives in this bin
        pos_mask = bin_labels .== 1
        neg_mask = bin_labels .== 0
        n_pos = sum(pos_mask)
        n_neg = sum(neg_mask)

        n_pos_bins[bin] = n_pos
        n_neg_bins[bin] = n_neg

        # Check if we have enough samples for reliable calculation
        if n_pos < min_bin_samples || n_neg < min_bin_samples
            efficiencies[bin] = NaN
            efficiency_errors[bin] = NaN
            thresholds[bin] = NaN
            achieved_misids[bin] = NaN
            misid_errors[bin] = NaN
            continue
        end

        # Find threshold for the target misID rate in this bin
        bin_workingpoints =
            find_workingpoints(bin_scores, bin_labels, misid_rates = [misid_rate])
        wp = bin_workingpoints[misid_rate]

        # Store results
        thresholds[bin] = wp.threshold
        achieved_misids[bin] = wp.achieved_misid
        efficiencies[bin] = wp.efficiency

        # Calculate errors
        # Binomial error for efficiency: √(p(1-p)/n)
        efficiency_errors[bin] = sqrt((wp.efficiency * (1 - wp.efficiency)) / n_pos)

        # Binomial error for misID rate
        misid_errors[bin] = sqrt((wp.achieved_misid * (1 - wp.achieved_misid)) / n_neg)
    end

    return (
        bin_centers = bin_centers,
        efficiency = efficiencies,
        efficiency_error = efficiency_errors,
        thresholds = thresholds,
        achieved_misid = achieved_misids,
        misid_error = misid_errors,
        n_pos = n_pos_bins,
        n_neg = n_neg_bins,
    )
end

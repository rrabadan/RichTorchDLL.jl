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

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

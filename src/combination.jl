function combination_model(; init_w = nothing, init_b = nothing)
    model = Dense(2 => 1)
    if !isnothing(init_w)
        model.weight .= init_w
    end
    if !isnothing(init_b)
        model.bias .= init_b
    end
    return model
end

function constrained_combination_model(; init_w2 = 0.1f0, init_b = 0.0f0)
    model = combination_model(init_w = [1.0f0, init_w2], init_b = [init_b])

    # Create a wrapper to ensure first weight stays fixed
    function forward(X)
        model.weight[1, 1] = 1.0f0 # reset w1 to 1.0
        return model(X)
    end
    return (model = model, forward = forward)
end

# Sigmoid function definition
σ(x) = 1 / (1 + exp(-x))

function pairwise_ranking_loss(scores, targets)
    pos_indices = findall(targets .== 1)
    neg_indices = findall(targets .== 0)

    if isempty(pos_indices) || isempty(neg_indices)
        return 0.0  # Handle edge case
    end

    # Extract scores
    pos_scores = scores[pos_indices]
    neg_scores = scores[neg_indices]

    # Compute all differences using broadcasting
    diffs = neg_scores .- permutedims(pos_scores)

    # Clip differences to prevent extreme values
    diffs_clipped = clamp.(diffs, -50.0, 50.0)

    # Apply sigmoid to all differences at once
    sig_diffs = σ.(diffs_clipped)

    # Check for NaNs
    if any(isnan, sig_diffs)
        println("WARNING: NaN in sigmoid output")
        println("Min diff: $(minimum(diffs)), Max diff: $(maximum(diffs))")
        return 0.0  # Return a default value instead of NaN
    end

    # Sum and normalize
    return sum(sig_diffs) / (length(pos_indices) * length(neg_indices))
end

function loss(model, X, y)
    scores = vec(model(X))
    return pairwise_ranking_loss(scores, y)
end

# AUC calculation (for evaluation)
function calculate_auc(scores, targets)
    pos = findall(targets .== 1)
    neg = findall(targets .== 0)

    # Count correct pairwise rankings
    correct = 0
    total = length(pos) * length(neg)

    for i in pos, j in neg
        if scores[i] > scores[j]
            correct += 1
        elseif scores[i] == scores[j]
            correct += 0.5  # Tie counts as half correct
        end
    end

    return correct / total
end

function pairwise_ranking_loss_sampled(scores, targets; max_pairs = 10000)
    pos_indices = findall(targets .== 1)
    neg_indices = findall(targets .== 0)

    if isempty(pos_indices) || isempty(neg_indices)
        return 0.0  # Handle edge case
    end

    # Determine total possible pairs
    total_pairs = length(pos_indices) * length(neg_indices)

    # If fewer than max_pairs, use all pairs
    if total_pairs <= max_pairs
        return pairwise_ranking_loss(scores, targets)
    end

    # Otherwise, sample pairs
    loss_sum = 0.0
    n_samples = min(max_pairs, total_pairs)

    for _ = 1:n_samples
        i = rand(pos_indices)
        j = rand(neg_indices)
        loss_sum += σ(scores[j] - scores[i])
    end

    return loss_sum / n_samples
end

# Memory-efficient AUC calculation using sampling
function calculate_auc_sampled(scores, targets; max_pairs = 10000)
    pos_indices = findall(targets .== 1)
    neg_indices = findall(targets .== 0)

    if isempty(pos_indices) || isempty(neg_indices)
        return 0.5  # Default for edge case
    end

    # Determine total possible pairs
    total_pairs = length(pos_indices) * length(neg_indices)

    # If fewer than max_pairs, use all pairs
    if total_pairs <= max_pairs
        return calculate_auc(scores, targets)
    end

    # Otherwise, sample pairs
    correct = 0.0
    n_samples = min(max_pairs, total_pairs)

    for _ = 1:n_samples
        i = rand(pos_indices)
        j = rand(neg_indices)
        if scores[i] > scores[j]
            correct += 1.0
        elseif scores[i] == scores[j]
            correct += 0.5
        end
    end

    return correct / n_samples
end


function threshold_auc_surrogate_loss(scores, targets)
    pos = findall(targets .== 1)
    neg = findall(targets .== 0)
    loss = 0.0

    # Hinge-like losses for threshold violations
    threshold_weight = 2.0  # Weight for threshold violations

    for i in pos
        # Penalize positive samples with score <= 0
        if scores[i] <= 0
            loss += threshold_weight * σ(-scores[i])
        end
    end

    for j in neg
        # Penalize negative samples with score > 0
        if scores[j] > 0
            loss += threshold_weight * σ(scores[j])
        end
    end

    # Also include ranking component (original pairwise loss)
    for i in pos, j in neg
        loss += σ(scores[j] - scores[i])
    end

    # Normalize loss
    total_pairs = length(pos) * length(neg)
    total_samples = length(pos) + length(neg)
    normalization = total_pairs + threshold_weight * total_samples

    return loss / normalization
end

# Stratified sampled AUC estimator (stable under class imbalance)
"""
    calculate_auc_stratified_sampled(scores, targets; max_pairs=10000, repeats=1)

Estimate AUC by sampling a small stratified set of positives and negatives. This
ensures the minority class is represented and keeps the total number of pairs
bounded by `max_pairs`. If `repeats>1` the function returns a NamedTuple with
mean and std over repeats.
"""
function calculate_auc_stratified_sampled(
    scores::AbstractVector{<:Real},
    targets::AbstractVector{<:Integer};
    max_pairs::Integer = 10000,
    repeats::Integer = 1,
)
    @assert length(scores) == length(targets)
    pos = findall(targets .== 1)
    neg = findall(targets .== 0)
    npos = length(pos)
    nneg = length(neg)
    if npos == 0 || nneg == 0
        return 0.5
    end

    aucs = Float64[]
    for _ = 1:repeats
        # choose stratified sample sizes
        Spos = min(npos, ceil(Int, sqrt(max_pairs)))
        Sneg = min(nneg, max(1, floor(Int, max_pairs ÷ max(1, Spos))))

        sampled_pos = rand(pos, Spos)
        sampled_neg = rand(neg, Sneg)

        pos_scores = scores[sampled_pos]
        neg_scores = scores[sampled_neg]

        # compute pairwise comparisons (Spos x Sneg)
        gt = sum(pos_scores .> permutedims(neg_scores))
        eq = sum(pos_scores .== permutedims(neg_scores))
        total_pairs = Spos * Sneg
        auc_est = (gt + 0.5 * eq) / total_pairs
        push!(aucs, auc_est)
    end

    if repeats == 1
        return first(aucs)
    else
        return (mean = mean(aucs), std = std(aucs))
    end
end

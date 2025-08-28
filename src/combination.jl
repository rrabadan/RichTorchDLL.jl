"""
    combination_model(; init_w=nothing, init_b=nothing)

Creates and returns a combination model, optionally initializing the model's weights (`init_w`) and biases (`init_b`).

# Arguments
- `init_w`: Optional initial weights for the model. Default is `nothing`.
- `init_b`: Optional initial biases for the model. Default is `nothing`.

# Returns
A combination model instance with the specified or default initialization.
"""
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

"""
    constrained_combination_model(; init_w2=0.1f0, init_b=0.0f0)

Creates a constrained combination model with optional initial values for parameters.

# Keyword Arguments
- `init_w2::Float32`: Initial value for the weight parameter `w2`. Defaults to `0.1f0`.
- `init_b::Float32`: Initial value for the bias parameter `b`. Defaults to `0.0f0`.

# Returns
A model object with the specified initial parameters.
"""
function constrained_combination_model(; init_w2 = 0.1f0, init_b = 0.0f0)
    model = combination_model(init_w = [1.0f0, init_w2], init_b = [init_b])

    # Create a wrapper to ensure first weight stays fixed
    function forward(X)
        model.weight[1, 1] = 1.0f0 # reset w1 to 1.0
        return model(X)
    end
    return (model = model, forward = forward)
end

# AUC calculation (for evaluation)
"""
    calculate_auc(scores, targets)

Calculates the Area Under the Curve (AUC) for a set of prediction scores and corresponding target labels.

# Arguments
- `scores`: A vector of predicted scores or probabilities for each sample.
- `targets`: A vector of true binary labels (typically 0 or 1) for each sample.

# Returns
- The AUC value as a `Float64`, representing the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.
"""
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

# Memory-efficient AUC calculation using sampling
"""
    calculate_auc_sampled(scores, targets; max_pairs=10000)

Estimates the Area Under the Curve (AUC) for a set of prediction `scores` and corresponding binary `targets` using random sampling.

# Arguments
- `scores`: A vector of predicted scores (higher means more likely positive).
- `targets`: A vector of true binary labels (0 or 1), same length as `scores`.
- `max_pairs`: (Optional) Maximum number of positive-negative pairs to sample for AUC estimation. Default is 10,000.

# Returns
- Estimated AUC value as a `Float64`.

# Notes
- This function is useful for large datasets where computing the exact AUC is computationally expensive.
- The result is an approximation based on randomly sampled positive-negative pairs.
"""
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

# Stratified sampled AUC estimator (stable under class imbalance)
"""
    calculate_auc_stratified_sampled(
        scores::AbstractVector{<:Real},
        targets::AbstractVector{<:Integer};
        max_pairs::Integer = 10000,
        repeats::Integer = 1,
    )

Compute the Area Under the Curve (AUC) for a set of scores and binary targets using stratified random sampling.

# Arguments
- `scores`: A vector of predicted scores (higher means more likely positive).
- `targets`: A vector of integer target labels (0 for negative, 1 for positive).
- `max_pairs`: Maximum number of positive-negative pairs to sample for AUC estimation (default: 10,000).
- `repeats`: Number of times to repeat the sampling procedure for averaging (default: 1).

# Returns
- The estimated AUC as a `Float64` value.

# Notes
- This function is useful for large datasets where computing the full AUC is computationally expensive.
- The result is an approximation of the true AUC, with accuracy depending on `max_pairs` and `repeats`.
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


function auc_with_threshold_penalty(scores, targets)
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

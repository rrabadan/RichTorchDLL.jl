using Flux: train!, Descent
using Plots
using Printf
using DataFrames

using Statistics: mean

# Function to prepare data for training
function prepare_data(rich, torch, labels)
    # Combine rich and torch into a single feature matrix
    X = transpose(hcat(rich, torch))

    # Convert to Float32
    X = Float32.(X)
    y = labels

    return X, y
end

function train_model!(
    model,
    X,
    y;
    epochs = 100,
    lr = 0.01,
    batch_size = 64,
    shuffle = true,
    verbose = true,
    plot_progress = false,
    save_path = nothing,
    eval_frequency = 1,     # How often to evaluate on full dataset
    max_eval_samples = 5000, # Maximum samples to use for evaluation
)
    # Setup optimizer
    opt = Descent(lr)
    opt_state = Flux.setup(opt, model)

    # Create DataLoader for batching
    loader = Flux.DataLoader((X, y), batchsize = batch_size, shuffle = shuffle)

    # For storing results
    results = DataFrame(
        epoch = Int[],
        batch_loss = Float64[],
        full_loss = Float64[],
        auc = Float64[],
    )

    # Create evaluation subset for memory efficiency
    n_samples = size(X, 2)
    if n_samples > max_eval_samples
        eval_indices = rand(1:n_samples, max_eval_samples)
        X_eval = X[:, eval_indices]
        y_eval = y[eval_indices]
    else
        X_eval = X
        y_eval = y
    end

    for epoch = 1:epochs
        # Track batch losses for this epoch
        batch_losses = Float64[]

        # Train on batches
        for (x_batch, y_batch) in loader
            # Compute loss and gradients
            loss_val, grads = Flux.withgradient(model) do m
                scores = vec(m(x_batch))
                # Use batch version that's more memory efficient
                pairwise_ranking_loss(scores, y_batch)
            end

            # Update model parameters
            Flux.update!(opt_state, model, grads[1])

            # Store batch loss
            push!(batch_losses, loss_val)
        end

        avg_batch_loss = mean(batch_losses)

        # Only evaluate on full dataset periodically to save memory
        if epoch == 1 || epoch % eval_frequency == 0 || epoch == epochs
            # Use evaluation subset
            eval_scores = vec(model(X_eval))
            eval_loss =
                pairwise_ranking_loss_sampled(eval_scores, y_eval, max_pairs = Int(1e6))
            eval_auc =
                calculate_auc_stratified_sampled(eval_scores, y_eval, max_pairs = Int(1e6))

            push!(results, (epoch, avg_batch_loss, eval_loss, eval_auc))

            # Print progress
            if verbose
                @printf(
                    "Epoch %4d, Batch Loss: %0.6f, Eval Loss: %0.6f, Eval AUC: %0.6f\n",
                    epoch,
                    avg_batch_loss,
                    eval_loss,
                    eval_auc
                )
            end
        else
            # Just record batch loss for non-evaluation epochs
            push!(results, (epoch, avg_batch_loss, NaN, NaN))

            if verbose
                @printf("Epoch %4d, Batch Loss: %0.6f\n", epoch, avg_batch_loss)
            end
        end
    end

    # Plot training progress
    if plot_progress
        p = plot(
            results.epoch,
            results.batch_loss,
            label = "Batch Loss",
            ylabel = "Loss",
            xlabel = "Epoch",
            legend = :topright,
        )

        # Plot evaluation metrics when available
        eval_epochs = findall(!isnan, results.auc)
        if !isempty(eval_epochs)
            p2 = plot(
                results.epoch[eval_epochs],
                results.auc[eval_epochs],
                label = "Eval AUC",
                color = :red,
                ylabel = "AUC",
                xlabel = "Epoch",
            )
            final_plot = plot(p, p2, layout = (2, 1), size = (800, 600))
        else
            final_plot = p
        end

        display(final_plot)

        # Save plot if path provided
        if !isnothing(save_path)
            savefig(final_plot, save_path)
        end
    end

    # Print final results
    final_row = results[end, :]
    println("\nTraining completed:")
    println("Final batch loss: $(final_row.batch_loss)")

    if !isnan(final_row.full_loss)
        println("Final evaluation loss: $(final_row.full_loss)")
        println("Final evaluation AUC: $(final_row.auc)")
    end

    println("Model parameters: W = $(model.weight), b = $(model.bias)")

    # Return results DataFrame
    return results
end

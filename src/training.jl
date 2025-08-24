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
    epochs=100,
    lr=0.01,
    batch_size=64,
    shuffle=true,
    verbose=true,
    plot_progress=false,
    save_path=nothing,
)
    # Setup optimizer
    opt = Descent(lr)
    opt_state = Flux.setup(opt, model)

    # Create DataLoader for batching
    loader = Flux.DataLoader((X, y), batchsize=batch_size, shuffle=shuffle)

    # For storing results
    epoch_losses = Float64[]
    epoch_aucs = Float64[]
    batch_losses = Float64[]

    # Progress tracking
    n_batches = length(loader)

    for epoch = 1:epochs
        # Reset batch losses for this epoch
        empty!(batch_losses)

        # Train on batches
        for (x_batch, y_batch) in loader
            # Compute loss and gradients
            loss_val, grads = Flux.withgradient(model) do m
                scores = vec(m(x_batch))
                pairwise_ranking_loss(scores, y_batch)
            end

            # Update model parameters
            Flux.update!(opt_state, model, grads[1])

            # Store batch loss
            push!(batch_losses, loss_val)
        end

        # Calculate metrics on full dataset for this epoch
        full_scores = vec(model(X))
        epoch_loss = pairwise_ranking_loss(full_scores, y)
        epoch_auc = calculate_auc(full_scores, y)

        push!(epoch_losses, epoch_loss)
        push!(epoch_aucs, epoch_auc)

        # Print progress
        if verbose && (epoch == 1 || epoch % 10 == 0 || epoch == epochs)
            avg_batch_loss = mean(batch_losses)
            @printf(
                "Epoch %4d, Batch Loss: %0.6f, Full Loss: %0.6f, AUC: %0.6f\n",
                epoch,
                avg_batch_loss,
                epoch_loss,
                epoch_auc
            )
        end
    end

    # Create results DataFrame
    results = DataFrame(epoch=1:epochs, loss=epoch_losses, auc=epoch_aucs)

    # Plot training progress
    if plot_progress
        p = plot(
            results.epoch,
            results.loss,
            label="Loss",
            ylabel="Loss",
            xlabel="Epoch",
            legend=:topright,
        )
        p2 = plot(
            results.epoch,
            results.auc,
            label="AUC",
            color=:red,
            ylabel="AUC",
            xlabel="Epoch",
        )
        final_plot = plot(p, p2, layout=(2, 1), size=(800, 600))
        display(final_plot)

        # Save plot if path provided
        if !isnothing(save_path)
            savefig(final_plot, save_path)
        end
    end

    # Print final results
    final_loss = epoch_losses[end]
    final_auc = epoch_aucs[end]
    println("\nTraining completed:")
    println("Final loss: $final_loss")
    println("Final AUC: $final_auc")

    println("Model parameters: W = $(model.weight), b = $(model.bias)")

    # Return results DataFrame
    return results
end

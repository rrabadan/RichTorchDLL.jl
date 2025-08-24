using Flux: train!, Descent
using Plots
using Printf
using DataFrames

function train_model!(model, X, y;
    epochs=100,
    lr=0.01,
    verbose=true,
    plot_progress=false,
    save_path=nothing
)
    opt = Descent(lr)

    # Initialize tracking arrays
    losses = []
    aucs = []
    epochs_arr = []

    # Training loop
    for epoch = 1:epochs
        # Train for one epoch
        train!(loss, model, [(X, y)], opt)

        # Calculate and track metrics
        current_loss = loss(model, X, y)
        current_scores = model(X)[:]
        current_auc = calculate_auc(current_scores, y)

        push!(losses, current_loss)
        push!(aucs, current_auc)
        push!(epochs_arr, epoch)

        # Print progress
        if verbose && (epoch == 1 || epoch % 10 == 0 || epoch == epochs)
            @printf("Epoch %4d, Loss: %0.6f, AUC: %0.6f\n", epoch, current_loss, current_auc)
        end
    end

    # Create result DataFrame
    results = DataFrame(
        epoch=epochs_arr,
        loss=losses,
        auc=aucs
    )

    # Plot training progress
    if plot_progress
        p = plot(results.epoch, results.loss, label="Loss", ylabel="Loss", xlabel="Epoch", legend=:topright)
        p2 = plot(results.epoch, results.auc, label="AUC", color=:red, ylabel="AUC", xlabel="Epoch")
        final_plot = plot(p, p2, layout=(2, 1), size=(800, 600))
        display(final_plot)

        # Save plot if path provided
        if !isnothing(save_path)
            savefig(final_plot, save_path)
        end
    end

    # Print final results
    final_auc = aucs[end]
    final_loss = losses[end]
    println("\nTraining completed:")
    println("Final loss: $final_loss")
    println("Final AUC: $final_auc")

    # Extract model parameters
    println("Model parameters: W = $(model.weight), b = $(model.bias)")

    # Return results DataFrame
    return results
end

# Function to prepare data for training
function prepare_data(rich, torch, labels)
    # Combine rich and torch into a single feature matrix
    X = transpose(hcat(rich, torch))

    # Convert to Float32
    X = Float32.(X)
    y = labels

    return X, y
end

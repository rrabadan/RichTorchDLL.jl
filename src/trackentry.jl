using CairoMakie: Figure, Axis, hist!, heatmap!, Colorbar

function create_trackentry_histogram2d(
    x,
    y;
    nbinsx = 100,
    nbinsy = 100,
    xrange = nothing,
    yrange = nothing,
)

    if xrange == nothing || yrange == nothing
        # Define bin edges
        xmin, xmax = extrema(x)
        ymin, ymax = extrema(y)

        # Add a small margin
        xmargin = 0.05 * (xmax - xmin)
        ymargin = 0.05 * (ymax - ymin)

        xedges = range(xmin - xmargin, xmax + xmargin, length = nbinsx + 1)
        yedges = range(ymin - ymargin, ymax + ymargin, length = nbinsy + 1)
    else
        xedges = range(xrange[1], xrange[2], length = nbinsx + 1)
        yedges = range(yrange[1], yrange[2], length = nbinsy + 1)
    end

    # Initialize count matrix
    counts = zeros(Int, nbinsx, nbinsy)

    # Bin the data
    for i = 1:length(x)
        ix = searchsortedfirst(xedges, x[i]) - 1
        iy = searchsortedfirst(yedges, y[i]) - 1

        # Check bounds
        if 1 <= ix <= nbinsx && 1 <= iy <= nbinsy
            counts[ix, iy] += 1
        end
    end

    # Return bin centers and counts
    xcenters = [(xedges[i] + xedges[i+1]) / 2 for i = 1:nbinsx]
    ycenters = [(yedges[i] + yedges[i+1]) / 2 for i = 1:nbinsy]

    return xcenters, ycenters, counts
end

function trackentry_heatmap(
    x,
    y;
    nbinsx = 100,
    nbinsy = 100,
    xrange = (-3000, 3000),
    yrange = (-2550, 2550),
    figsize = (1000, 600),
)
    valid_mask = .!isnan.(x) .& .!isnan.(y) .& .!isinf.(x) .& .!isinf.(y)

    # Count how many invalid values we found
    invalid_count = length(x) - sum(valid_mask)
    if invalid_count > 0
        println("Filtered out $invalid_count invalid (NaN/Inf) momentum values")
    end

    # Extract valid momentum values
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # Now bin the filtered momentum data
    x_bins, y_bins, counts = create_trackentry_histogram2d(
        x_valid,
        y_valid,
        nbinsx = nbinsx,
        nbinsy = nbinsy,
        xrange = xrange,
        yrange = yrange,
    )

    # Create the figure and axis
    fig = Figure(size = figsize)
    ax = Axis(
        fig[1, 1],
        title = "TORCH Track Entry",
        xlabel = "X [mm]",
        ylabel = "Y [mm]",
        xticklabelsize = 18,
        yticklabelsize = 18,
        xlabelsize = 20,
        ylabelsize = 20,
        titlesize = 20,
    )

    # Custom colormap where the first color is white
    # and the rest follows viridis/turbo
    custom_cmap = vcat([:white], collect(cgrad(:viridis, 100)))

    # Use this custom colormap in your heatmap
    hm = heatmap!(ax, x_bins, y_bins, counts, colormap = custom_cmap, lowclip = :white)  # Make sure the lowest values also appear white

    # hm_entry = heatmap!(ax_entry, x_bins, y_bins, counts', 
    #            colormap = :viridis)

    # Add colorbar
    Colorbar(fig[1, 2], hm, label = "Counts")

    # Display in notebook (might work)
    fig
end

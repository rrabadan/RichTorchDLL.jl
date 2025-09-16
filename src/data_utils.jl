using DataFrames
using UnROOT

"""
    load_data(data_dir, luminosity, scenario)

Load data from ROOT file for a given luminosity and scenario.

# Arguments
- `data_dir`: Directory containing input ROOT files
- `luminosity`: Peak luminosity (Default or Medium)
- `scenario`: Scenario name (baseline or middle)

# Returns
A DataFrame containing the loaded data
"""
function load_data(
    data_dir::String,
    luminosity::String,
    scenario::String,
    no_central_modules::Bool,
)
    filename = "Expert-ProtoTuple-Run5-$(luminosity)-$(scenario).root"
    if no_central_modules
        filename = replace(filename, ".root" => "-nocentralmod.root")
        println("Loading data without central modules")
    end
    filepath = joinpath(data_dir, filename)
    println("Loading data from: $filepath")

    if !isfile(filepath)
        println("File not found: $filepath")
    end

    f = ROOTFile(filepath)

    # Define which branches to read
    branches = [
        "RichDLLk",
        "RichDLLp",
        "TorchDLLk",
        "TorchDLLp",      # DLL variables
        "MCParticleType", # True particle ID
        "TrackP",         # Momentum
        "TrackPt",         # Momentum
        "TrackType",
        "TrackChi2PerDof",
        "RichUsedR2Gas",
        "RichUsedR1Gas",
    ]

    # Read tree
    t = LazyTree(f, "ChargedProtoTuple/protoPtuple", branches)
    df = DataFrame(t)
    println(nrow(df), " entries loaded from ", filepath)

    return df
end

"""
    load_data_pair(data_dir, luminosity)

Load data from ROOT files for a given luminosity, both baseline and middle scenarios.

# Arguments
- `data_dir`: Directory containing input ROOT files
- `luminosity`: Peak luminosity (Default or Medium)

# Returns
A tuple of DataFrames (baseline_df, middle_df)
"""
function load_data_pair(data_dir::String, luminosity::String)
    filename_base = "Expert-ProtoTuple-Run5-$(luminosity)-baseline.root"
    filepath_base = joinpath(data_dir, filename_base)

    filename_mid = "Expert-ProtoTuple-Run5-$(luminosity)-middle.root"
    filepath_mid = joinpath(data_dir, filename_mid)

    if !isfile(filepath_base)
        error("File not found: $filepath_base")
    end
    if !isfile(filepath_mid)
        error("File not found: $filepath_mid")
    end

    f_base = ROOTFile(filepath_base)
    f_mid = ROOTFile(filepath_mid)

    # Define which branches to read
    branches = [
        "RichDLLk",
        "RichDLLp",
        "TorchDLLk",
        "TorchDLLp",     # DLL variables
        "MCParticleType", # True particle ID
        "TrackP",         # Momentum
    ]

    # Read trees
    t_base = LazyTree(f_base, "ChargedProtoTuple/protoPtuple", branches)
    df_base = DataFrame(t_base)
    println(nrow(df_base), " entries loaded from ", filepath_base)

    t_mid = LazyTree(f_mid, "ChargedProtoTuple/protoPtuple", branches)
    df_mid = DataFrame(t_mid)
    println(nrow(df_mid), " entries loaded from ", filepath_mid)

    return df_base, df_mid
end

function is_kaon(id::Integer)
    return abs(id) == 321
end

function is_pion(id::Integer)
    return abs(id) == 211
end

function is_proton(id::Integer)
    return abs(id) == 2212
end

"""
    filter_by_particle_type(df, particle_types)

Filter DataFrame to include only rows with MCParticleType in the specified list.

# Arguments
- `df`: DataFrame to filter
- `particle_types`: List of functions like `is_kaon`, `is_pion`, `is_proton`

# Returns
A filtered DataFrame
"""
function filter_by_particle_type(df::DataFrame, particle_types::Vector{<:Function})
    return filter(row -> any(f -> f(row.MCParticleType), particle_types), df)
end

function filter_by_particle_type!(df::DataFrame, particle_types::Vector{<:Function})
    filter!(row -> any(f -> f(row.MCParticleType), particle_types), df)
end


"""
    filter_kaon_pion(df)

Filter DataFrame to include only kaons and pions.

# Arguments
- `df`: DataFrame to filter

# Returns
A filtered DataFrame
"""
function filter_kaon_pion(df::DataFrame)
    return filter_by_particle_type(df, [is_kaon, is_pion])
end

function filter_kaon_pion!(df::DataFrame)
    filter_by_particle_type!(df, [is_kaon, is_pion])
end

"""
    filter_proton_kaon(df)

Filter DataFrame to include only protons and kaons.

# Arguments
- `df`: DataFrame to filter

# Returns
A filtered DataFrame
"""
function filter_proton_kaon(df::DataFrame)
    return filter_by_particle_type(df, [is_proton, is_kaon])
end

function filter_proton_kaon!(df::DataFrame)
    filter_by_particle_type!(df, [is_proton, is_kaon])
end

"""
    filter_proton_pion(df)

Filter DataFrame to include only protons and pions.

# Arguments
- `df`: DataFrame to filter

# Returns
A filtered DataFrame
"""
function filter_proton_pion(df::DataFrame)
    return filter_by_particle_type(df, [is_proton, is_pion])
end

function filter_proton_pion!(df::DataFrame)
    filter_by_particle_type!(df, [is_proton, is_pion])
end

"""
    filter_by_momentum(df, min_p, max_p)

Filter DataFrame to include only tracks with momentum in [min_p, max_p] MeV/c.

# Arguments
- `df`: DataFrame to filter
- `min_p`: Minimum momentum in MeV/c
- `max_p`: Maximum momentum in MeV/c

# Returns
A filtered DataFrame
"""
function filter_by_momentum(df::DataFrame, min_p::Real, max_p::Real)
    return filter(:TrackP => p -> min_p < p < max_p, df)
end

function filter_by_momentum!(df::DataFrame, min_p::Real, max_p::Real)
    filter!(:TrackP => p -> min_p < p < max_p, df)
end

function filter_by_pt(df::DataFrame, min_pt::Real, max_pt::Real)
    return filter(:TrackPt => pt -> min_pt < pt < max_pt, df)
end

function filter_by_pt!(df::DataFrame, min_pt::Real, max_pt::Real)
    filter!(:TrackPt => pt -> min_pt < pt < max_pt, df)
end

"""
    filter_by_dll_range(df, min_dll, max_dll; dll_columns=["RichDLLk", "RichDLLp", "TorchDLLk", "TorchDLLp"])

Filter DataFrame to include only tracks with DLL values in [min_dll, max_dll].

# Arguments
- `df`: DataFrame to filter
- `min_dll`: Minimum DLL value
- `max_dll`: Maximum DLL value
- `dll_columns`: List of DLL column names to filter. Default includes both RICH and TORCH DLLs.

# Returns
A filtered DataFrame
"""
function filter_by_dll_range(
    df::DataFrame,
    min_dll::Real,
    max_dll::Real;
    dll_columns = ["RichDLLk", "RichDLLp", "TorchDLLk", "TorchDLLp"],
)
    # Check which of the specified DLL columns exist in the DataFrame
    valid_columns = []
    for col in dll_columns
        if col in names(df)
            push!(valid_columns, col)
        end
    end

    if isempty(valid_columns)
        error("None of the specified DLL columns found in DataFrame")
    end

    # Apply filter to each DLL column
    filtered_df = df
    for col in valid_columns
        filtered_df = filter(col => dll -> min_dll < dll < max_dll, filtered_df)
    end

    return filtered_df
end

function filter_by_dll_range!(
    df::DataFrame,
    min_dll::Real,
    max_dll::Real;
    dll_columns = ["RichDLLk", "RichDLLp", "TorchDLLk", "TorchDLLp"],
)
    # Check which of the specified DLL columns exist in the DataFrame
    valid_columns = []
    for col in dll_columns
        if col in names(df)
            push!(valid_columns, col)
        end
    end

    if isempty(valid_columns)
        error("None of the specified DLL columns found in DataFrame")
    end

    # Apply filter to each DLL column
    for col in valid_columns
        filter!(col => dll -> min_dll < dll < max_dll, df)
    end

    return df
end

"""
    filter_valid_dll(df; dll_columns=["RichDLLk", "RichDLLp", "TorchDLLk", "TorchDLLp"])

Filter DataFrame to include only rows with non-zero DLL values for the specified columns.

# Arguments
- `df`: DataFrame to filter
- `dll_columns`: List of DLL column names to check for non-zero values. 
                 Default includes both RICH and TORCH DLLs.

# Returns
A filtered DataFrame with valid DLL values for all specified columns
"""
function filter_valid_dll(
    df::DataFrame;
    dll_columns = ["RichDLLk", "RichDLLp", "TorchDLLk", "TorchDLLp"],
)
    # Check which of the specified DLL columns exist in the DataFrame
    valid_columns = []
    for col in dll_columns
        if col in names(df)
            push!(valid_columns, col)
        end
    end

    if isempty(valid_columns)
        error("None of the specified DLL columns found in DataFrame")
    end

    # Filter out rows with zero DLL values
    if length(valid_columns) == 1
        col = valid_columns[1]
        return filter(col => v -> v != 0.0, df)
    else
        # Apply sequential filtering for each column instead of using a tuple
        filtered_df = df
        for col in valid_columns
            filtered_df = filter(col => v -> v != 0.0, filtered_df)
        end
        return filtered_df
    end
end

function filter_valid_dll!(
    df::DataFrame;
    dll_columns = ["RichDLLk", "RichDLLp", "TorchDLLk", "TorchDLLp"],
)
    # Check which of the specified DLL columns exist in the DataFrame
    valid_columns = []
    for col in dll_columns
        if col in names(df)
            push!(valid_columns, col)
        end
    end

    if isempty(valid_columns)
        error("None of the specified DLL columns found in DataFrame")
    end

    # Filter out rows with zero DLL values
    if length(valid_columns) == 1
        col = valid_columns[1]
        filter!(col => v -> v != 0.0, df)
    else
        # Apply sequential filtering for each column instead of using a tuple
        for col in valid_columns
            filter!(col => v -> v != 0.0, df)
        end
    end
    return df
end

"""
    filter_valid_torch(df)

Filter DataFrame to include only rows with non-zero TORCH DLL values.
This is a convenience wrapper around filter_valid_dll.

# Arguments
- `df`: DataFrame to filter

# Returns
A filtered DataFrame with valid TORCH DLL values
"""
function filter_valid_torch(df::DataFrame)
    return filter_valid_dll(df, dll_columns = ["TorchDLLk", "TorchDLLp"])
end

"""
    filter_valid_rich(df)

Filter DataFrame to include only rows with non-zero RICH DLL values.
This is a convenience wrapper around filter_valid_dll.

# Arguments
- `df`: DataFrame to filter

# Returns
A filtered DataFrame with valid RICH DLL values
"""
function filter_valid_rich(df::DataFrame)
    #return filter_valid_dll(df, dll_columns=["RichDLLk", "RichDLLp"])
    fdf = filter([:TrackType, :TrackChi2PerDof] => (t, chi2) -> t == 3 && chi2 < 5, df)
    filter!([:RichUsedR1Gas, :RichUsedR2Gas] => (r1, r2) -> r1 == true && r2 == true, fdf)
    return fdf
end

"""
    prepare_dataset(df; 
                    particle_types = [is_kaon, is_pion],
                    min_p = 2000,
                    max_p = 15000,
                    min_pt = 500,
                    max_pt = 100000,
                    min_dll = -100,
                    max_dll = 100,
                    dll_columns = ["RichDLLk", "RichDLLp", "TorchDLLk", "TorchDLLp"])

Prepare a dataset by applying standard filtering steps.

# Arguments
- `df`: DataFrame to filter
- `particle_types`: List of functions like `is_kaon`, `is_pion`, `is_proton`
- `min_p`: Minimum momentum in MeV/c
- `max_p`: Maximum momentum in MeV/c
- `min_pt`: Minimum transverse momentum in MeV/c
- `max_pt`: Maximum transverse momentum in MeV/c
- `min_dll`: Minimum DLL value
- `max_dll`: Maximum DLL value
- `dll_columns`: List of DLL column names to filter. Default includes both RICH and TORCH DLLs.

# Returns
A named tuple with the filtered DataFrames:
- `filtered`: Main filtered DataFrame
- `torch`: Filtered DataFrame with valid TORCH DLL
- `rich`: Filtered DataFrame with valid RICH DLL
- `all_valid`: Filtered DataFrame with all DLLs valid
"""
function prepare_dataset(
    df::DataFrame;
    particle_types = [is_kaon, is_pion],
    min_p = 2000,
    max_p = 200000,
    min_pt = 500,
    max_pt = 100000,
    min_dll = -200,
    max_dll = 200,
    dlls = ["DLLk", "DLLp"],
)
    dll_columns = String[]
    dll_torch_columns = String[]
    dll_rich_columns = String[]
    for dll in dlls
        push!(dll_rich_columns, "Rich$(dll)")
        push!(dll_torch_columns, "Torch$(dll)")
    end
    dll_columns = vcat(dll_rich_columns, dll_torch_columns)

    # Filter by particle type
    filtered = filter_by_particle_type(df, particle_types)
    println("$(nrow(filtered)) entries after filtering for specified particle types")

    # Filter by momentum
    filtered = filter_by_momentum(filtered, min_p, max_p)
    println(
        "$(nrow(filtered)) entries after filtering for tracks with momentum range [$(min_p/1000), $(max_p/1000)] GeV/c",
    )

    # Filter by momentum
    filtered = filter_by_pt(filtered, min_pt, max_pt)
    println(
        "$(nrow(filtered)) entries after filtering for tracks with momentum range [$(min_pt/1000), $(max_pt/1000)] GeV/c",
    )

    # Filter by DLL range
    filtered = filter_by_dll_range(filtered, min_dll, max_dll, dll_columns = dll_columns)
    println("$(nrow(filtered)) entries after filtering for DLL in [$(min_dll), $(max_dll)]")

    # Create versions with valid DLLs
    torch_filtered = filter_valid_dll(filtered, dll_columns = dll_torch_columns)
    println("$(nrow(torch_filtered)) entries after filtering for valid TORCH DLL")

    rich_filtered = filter_valid_rich(filtered)
    println("$(nrow(rich_filtered)) entries after filtering for valid RICH DLL")

    all_valid = filter_valid_rich(torch_filtered)
    println("$(nrow(all_valid)) entries after filtering for all valid DLLs")

    return (
        filtered = filtered,
        torch = torch_filtered,
        rich = rich_filtered,
        all_valid = all_valid,
    )
end

"""
    create_binary_labels(df, positive_type_func)

Create binary labels where 1 corresponds to the positive class and 0 to all others.

# Arguments
- `df`: DataFrame containing particle information
- `positive_type_func`: Function that returns true for positive class (e.g., `is_kaon`)

# Returns
A vector of binary labels (0 or 1)
"""
function create_binary_labels(df::DataFrame, positive_type_func::Function)
    return Int.(positive_type_func.(df.MCParticleType))
end

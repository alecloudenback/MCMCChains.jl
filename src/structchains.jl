#################### Chains ####################


## Constructors ##

# Constructor to handle a vector of vectors.
Chains(val::AbstractVector{<:AbstractVector{<:Union{Missing, Real}}}, args...; kwargs...) =
    Chains(copy(reduce(hcat, val)'), args...; kwargs...)

# Constructor to handle a 1D array.
Samples(val::AbstractVector{<:Union{Missing, Real}}, args...; kwargs...) =
    Samples(reshape(val, :, 1, 1), args...; kwargs...)

# Constructor to handle a 2D array
Samples(val::AbstractMatrix{<:Union{Missing, Real}}, args...; kwargs...) =
    Samples(reshape(val, size(val, 1), size(val, 2), 1), args...; kwargs...)

# Constructor to handle parameter names that are not Symbols.
function Samples(
    val::AbstractArray{<:Union{Missing,Real},3},
    parameter_names::AbstractVector,
    args...;
    kwargs...
)
    return ChainArray(val, Symbol.(parameter_names), args...; kwargs...)
end

# Generic chain constructor.
# *must* be documented -- hard to understand how Chains work otherwise.
function ChainArray(
        val::AbstractArray{R,3},
        parameter_names::AbstractVector{Symbol} = Symbol.(:param_, 1:size(val, 2)),
        name_map = (parameters = parameter_names,);
        evidence = missing,
        info::NamedTuple = NamedTuple()
    )   where R <: Union{Missing, Real}
    return 
end

function ChainArray(
    val::AbstractArray{}
)
    # Check that iteration numbers are reasonable
    if length(iterations) ≠ size(val, 1)
        error("length of `iterations` (", length(iterations),
              ") is not equal to the number of iterations (", size(val, 1), ")")
    elseif !isempty(iterations) && first(iterations) < 1
        error("iteration numbers must be positive integers")
    end
    if !isstrictlyincreasing(iterations)
        error("iteration numbers must be strictly increasing")
    end

    # Make sure that we have a `:parameters` index and # Copying can avoid state mutation.
    _name_map = initnamemap(name_map)

    # Preclean the name_map of names that aren't in the parameter_names vector.
    for names in _name_map
        filter!(x -> x ∈ parameter_names, names)
    end

    # Store unassigned variables.
    unassigned = OrderedCollections.OrderedSet{Symbol}()

    # Check that all parameters are assigned.
    for param in parameter_names
        if all(param ∉ names for names in _name_map)
            push!(unassigned, param)
        end
    end

    # Assign all unassigned parameter names.
    append!(_name_map[:parameters], unassigned)

    # Construct the DimArray
    arr = DimArray(
        val, 
        (iter = iterations, var = parameter_names, chain = 1:size(val, 3),)
    )

    # Create the new chain.
    return Chains(arr, evidence, _name_map, info)
end

"""
    Chains(chains::Chains, section::Union{Symbol,String})
    Chains(chains::Chains, sections)

Return a new chain with only a specific `section` or multiple `sections` pulled out.

# Examples
```jldoctest
julia> chn = Chains(rand(100, 2, 1), [:a, :b], Dict(:internals => [:a]));

julia> names(chn)
2-element Vector{Symbol}:
 :a
 :b

julia> chn2 = Chains(chn, :internals);

julia> names(chn2)
1-element Vector{Symbol}:
 :a
```
"""
Chains(chains::Chains, section::Union{Symbol,String}) = Chains(chains, (section,))
function Chains(chn::Chains, sections)
    # Make sure the sections exist first.
	if any(!haskey(chn.name_map, Symbol(x)) for x in sections)
		error("some sections are not present in the chain")
    end

    # Create the new section map.
    name_map = (; (Symbol(k) => chn.name_map[Symbol(k)] for k in sections)...)

    # Extract wanted values.
    indices = reduce(vcat, name_map)
    value = chn.value[var=At(indices)]

    # Create the new chain.
    return Chains(value, chn.logevidence, name_map, chn.info)
end
Chains(chain::Chains, ::Nothing) = chain

# Groups of parameters

"""
    namesingroup(chains::Chains, sym::Union{String,Symbol})

Return the names of all parameters in a chain that belong to the group `sym`.

This is based on the MCMCChains convention that parameters with names of the form `:sym[index]`
belong to one group of parameters called `:sym`.

If the chain contains a parameter of name `:sym` it will be returned as well.

# Example
```jldoctest
julia> chn = Chains(rand(100, 2, 2), ["A[1]", "A[2]"]);

julia> namesingroup(chn, :A)
2-element Vector{Symbol}:
 Symbol("A[1]")
 Symbol("A[2]")
```
"""
namesingroup(chains::Chains, sym::String) = namesingroup(chains, Symbol(sym))
function namesingroup(chains::Chains, sym::Symbol)
    # Start by looking up the symbols in the list of parameter names.
    names_of_params = names(chains)
    regex = Regex("^$sym\$|^$sym\\[")
    indices = findall(x -> match(regex, string(x)) !== nothing, names(chains))
    return names_of_params[indices]
end

"""
    group(chains::Chains, name::Union{String,Symbol})

Return a subset of the chain chain with all parameters in the group `Symbol(name)`.
"""
function group(chains::Chains, name::Union{String,Symbol})
    return chains[:, namesingroup(chains, name), :]
end

#################### Indexing ####################

Base.getindex(chains::Chains, i::Integer) = chains.value[iter=i]
Base.getindex(chains::Chains, i::AbstractVector{<:Integer}) = chains.value[iter=i]
Base.getindex(chains::Chains, args...; kwargs...) = getindex(chains.value, args...; kwargs...)


Base.getindex(chains::Chains, v::Symbol) = chains.value[var=At(namesingroup(chains, v))]
Base.getindex(chains::Chains, v::AbstractVector{Symbol}) = chains.value[var=At(namesingroup(chains, v))]

Base.getindex(chains::Chains, v::String) = getindex(chains, Symbol(v))
Base.getindex(chains::Chains, v::AbstractVector{String}) = getindex(chains, Symbol.(v))


Base.getindex(chn::Chains, i, j, k) = _getindex(chn, chn.value[_toindex(i, j, k)...])

_getindex(::Chains, data) = data
function _getindex(chains::Chains, data::AbstractDimArray)
    names = dims(data, :var).val.data
    name_map = namemap_intersect(chains.name_map, names)
    return Chains(data, chains.logevidence, name_map, chains.info)
end

# convert strings to symbols but try to keep all dimensions for multiple parameters
_toindex(i, j, k) = (i, string2symbol(j), k)
_toindex(i::Integer, j, k) = (i:i, string2symbol(j), k)
_toindex(i, j, k::Integer) = (i, string2symbol(j), k:k)
_toindex(i::Integer, j, k::Integer) = (i:i, string2symbol(j), k:k)

# return an array or a number if a single parameter is specified
const SingleIndex = Union{Symbol,String,Integer}
_toindex(i, j::SingleIndex, k) = (i, string2symbol(j), k)
_toindex(i::Integer, j::SingleIndex, k) = (i, string2symbol(j), k)
_toindex(i, j::SingleIndex, k::Integer) = (i, string2symbol(j), k)
_toindex(i::Integer, j::SingleIndex, k::Integer) = (i, string2symbol(j), k)

Base.setindex!(chains::Chains, v, i...) = setindex!(chains.value, v, i...)
Base.lastindex(chains::Chains) = lastindex(chains.value, 1)
Base.lastindex(chains::Chains, d::Integer) = lastindex(chains.value, d)

"""
    Base.get(chains::Chains, key::Symbol; flatten=false)
    Base.get(chains::Chains, keys::Vector{Symbol}; flatten=false)

Return a `NamedTuple` with `key` as the key, and matching parameter names as the values.

Passing `flatten=true` will return a `NamedTuple` with keys ungrouped.

# Example

```jldoctest
julia> chn = Chains([1:2 3:4]);

julia> get(chn, :param_1)
(param_1 = [1; 2],)

julia> get(chn, [:param_1, "param_2"])
(param_2 = [3; 4],
 param_1 = [1; 2],)

julia> get(chn, :param_1; flatten=true)
(param_1 = 1,)
```
"""
Base.get(chains::Chains, keys; kwargs...) = get(chains, Symbol.(keys); kwargs...)
Base.get(chains::Chains, key::Symbol; kwargs...) = get(chains, [key]; kwargs...)
function Base.get(chains::Chains, keys::Vector{Symbol}; flatten = false)
    keys = [namesingroup(chains, key) |> _strip_singletons for key in keys]
    pairs = NamedTuple{keys}(Tuple([chains[key] for var in keys]))

    return pairs
end

"""
    get(chains::Chains; section::Union{Vector{Symbol}, Symbol; flatten=false}

Return all parameters in a given section(s) as a `NamedTuple`.

Passing `flatten=true` will return a `NamedTuple` with keys ungrouped.

# Example

```jldoctest
julia> chn = Chains([1:2 3:4], [:a, :b], Dict(:internals => [:a]));

julia> get(chn; section=:parameters)
(b = [3; 4],)

julia> get(chn; section=[:internals])
(a = [1; 2],)
```
"""
function Base.get(
    c::Chains;
    section::Union{Symbol,AbstractVector{Symbol}},
    flatten = false
)
    names = Set(Symbol[])
    regex = r"[^\[]*"
    _section = section isa Symbol ? (section,) : section
    for v in _section
        v in keys(c.name_map) || error("section $v does not exist")

        # If the name contains a bracket,
        # split it so get can group them correctly.
        if flatten
            append!(names, c.name_map[v])
        else
            for name in c.name_map[v]
                m = match(regex, string(name))
                push!(names, Symbol(m.match))
            end
        end
    end

    return get(c, collect(names); flatten = flatten)
end

"""
    get_params(chains::Chains; flatten=false)

Return all parameters packaged as a `NamedTuple`. Variables with a bracket
in their name (as in "P[1]") will be grouped into the returned value as P.

Passing `flatten=true` will return a `NamedTuple` with keys ungrouped.

# Example

```jldoctest
julia> chn = Chains(1:5);

julia> x = get_params(chn);

julia> x.param_1
5×1 DimArray{Int64,2} with dimensions: 
  Dim{:iter} Sampled 1:1:5 ForwardOrdered Regular Points,
  Dim{:chain} Sampled 1:1 ForwardOrdered Regular Points
and reference dimensions: 
  Dim{:var} Categorical Symbol[param_1] ForwardOrdered
 1
 2
 3
 4
 5
```
"""
get_params(c::Chains; flatten = false) = get(c, section = sections(c), flatten=flatten)

#################### Base Methods ####################

function Base.show(io::IO, chains::Chains)
    print(io, "MCMC chain (", summary(chains.value.data), ")")
end

function Base.show(io::IO, mime::MIME"text/plain", chains::Chains)
    print(io, "Chains ", chains, ":\n\n", header(chains))

    # Show summary stats.
    summaries = describe(chains)
    for summary in summaries
        println(io)
        show(io, mime, summary)
    end
end

Base.keys(chains::Chains) = names(chains)
Base.size(chains::Chains) = size(chains.value)
Base.size(chains::Chains, ind) = size(chains.value, ind)
Base.length(chains::Chains) = length(range(chains))
# Are these right? The original code returned the first/last index too, but that seems weird
Base.first(chains::Chains) = first(dims(chains.value, :iter).val.data)
Base.step(chains::Chains) = step(dims(chains.value, :iter).val.data)
Base.last(chains::Chains) = last(dims(chains.value, :iter).val.data)

Base.convert(::Type{Array}, chains::Chains) = convert(Array, chains.value)


#################### Timing Functions ####################

# Convenience functions to handle different types of timestamps.
to_datetime(t::DateTime) = t
to_datetime(t::Float64) = unix2datetime(t)
to_datetime(t) = missing_datetime(typeof(t))
to_datetime_vec(t::Union{Float64, DateTime}) = [to_datetime(t)]
to_datetime_vec(t::DateTime) = [to_datetime(t)]
to_datetime_vec(ts::Vector) = map(to_datetime, ts)
to_datetime_vec(ts) = [missing]

min_datetime(ts) = minimum(to_datetime_vec(ts))
max_datetime(ts) = maximum(to_datetime_vec(ts))

# does not specialize on `typeof(T)`
function missing_datetime(T::Type)
    @warn "timestamp of type $(T) unknown"
    return missing
end

"""
    min_start(chains::Chains)

Retrieve the minimum of the start times (as `DateTime`) from `chain.info`.

It is assumed that the start times are stored in `chain.info.start_time` as
`DateTime` or unix timestamps of type `Float64`.
"""
min_start(chains::Chains) = min_datetime(start_times(chains))

"""
    max_stop(chains::Chains)

Retrieve the maximum of the stop times (as `DateTime`) from `chain.info`.

It is assumed that the start times are stored in `chain.info.stop_time` as
`DateTime` or unix timestamps of type `Float64`.
"""
max_stop(chains::Chains) = max_datetime(stop_times(chains))

"""
    start_times(chains::Chains)

Retrieve the contents of `chains.info.start_time`, or `missing` if no 
`start_time` is set.
"""
start_times(chains::Chains) = to_datetime_vec(get(chains.info, :start_time, missing))

"""
    stop_times(chains::Chains)

Retrieve the contents of `chains.info.stop_time`, or `missing` if no 
`stop_time` is set.
"""
stop_times(chains::Chains) = to_datetime_vec(get(chains.info, :stop_time, missing))

"""
    wall_duration(chains::Chains; start=min_start(chains), stop=max_stop(chains))

Calculate the wall clock time for all chains in seconds.

The duration is calculated as `stop - start`, where as default `stop`
is the latest stopping time and `start` is the earliest starting time.
"""
function wall_duration(chains::Chains; start=min_start(chains), stop=max_stop(chains))
    # DateTime - DateTime returns a Millisecond value,
    # divide by 1k to get seconds.
    return if start === missing || stop === missing
        return missing
    else
        return Dates.value(stop - start) / 1000
    end
end

"""
    compute_duration(chains::Chains; start=start_times(chains), stop=stop_times(chains))

Calculate the compute time for all chains in seconds.

The duration is calculated as the sum of `start - stop` in seconds. 

`compute_duration` is more useful in cases of parallel sampling, where `wall_duration`
may understate how much computation time was utilitzed.
"""
function compute_duration(
    chains::Chains; 
    start=start_times(chains), 
    stop=stop_times(chains)
)
    # Calculate total time for each chain, then add it up.
    if start === missing || stop === missing
        return missing
    else
        calc = sum(stop - start)
        if calc === missing
            return missing
        else
            return Dates.value(calc) / 1000
        end
    end
end

#################### Auxilliary Functions ####################

"""
    dims(chains::Chains, args...; kwargs...)

Return the `Dim` structs used by `chains.value`. See `DimensionalData.dims` for additional
arguments.
"""
DimensionalData.dims(chains::Chains, args...; kwargs...) = dims(chains.value, args...; kwargs...)

"""
    range(chains::Chains)

Return the range of iteration indices of the `chains`.
"""
Base.range(chains::Chains) = dims(chains.value, :iter).val.data

"""
    setrange(chains::Chains, range::AbstractVector{Int})

Generate a new chain from `chains` with iterations indexed by `range`.

The new chain and `chains` share the same data in memory.
"""
function setrange(chains::Chains, range::AbstractVector{Int})
    if length(chains) != length(range)
        error("length of `range` (", length(range),
              ") is not equal to the number of iterations (", length(chains), ")")
    end
    if !isempty(range) && first(range) < 1
        error("iteration numbers must be positive integers")
    end

    value = DimArray(
        chains.value.data,
        (iter = range, var = names(chains), chain = MCMCChains.chains(chains))
    )

    return Chains(value, chains.logevidence, chains.name_map, chains.info)
end

"""
    resetrange(chains::Chains)

Generate a new chain from `chains` with iterations indexed by `1:n`, where `n` is the number
of samples per chain.

The new chain and `chains` share the same data in memory.
"""
resetrange(chains::Chains) = setrange(chains, 1:size(chains, 1))

"""
    chains(chains::Chains)

Return the names or symbols of each chain in a `Chains` object.
"""
chains(chains::Chains) = dims(chains.value, :chain).val.data

"""
    names(chains::Chains)

Return the parameter names in the `chains`.
"""
Base.names(chains::Chains) = dims(chains.value, :var).val.data

"""
    names(chains::Chains, section::Symbol)

Return the parameter names of a `section` in the `chains`.
"""
function Base.names(chains::Chains, section::Symbol)
    convert(Vector{Symbol}, chains.name_map[section])
end

"""
    names(chains::Chains, sections)

Return the parameter names of the `sections` in the `chains`.
"""
function Base.names(chains::Chains, sections)
    return [chains.name_map[section] for section in sections]
end

"""
    get_sections(chains[, sections])

Return multiple `Chains` objects, each containing only a single section.
"""
function get_sections(chains::Chains, sections = keys(chains.name_map))
    return [Chains(chains, section) for section in sections]
end
get_sections(chains::Chains, section::Union{Symbol, String}) = Chains(chains, section)

"""
    sections(chains::Chains)

Retrieve a list of the sections in a chain.
"""
sections(chains::Chains) = collect(keys(chains.name_map))

"""
    header(chains::Chains; section=missing)

Return a string containing summary information for a `Chains` object.
If the `section` keyword is used, this function prints only the relevant section
header.

# Example
```julia
# Printing the whole header.
header(chn)

# Print only one section's header.
header(chn, section = :parameter)
```
"""
function header(chains::Chains; section=missing)
    rng = range(chains)

    # Function to make section strings.
    section_str(sec, arr) = string(
        "$sec",
        repeat(" ", 18 - length(string(sec))),
        "= $(join(map(string, arr), ", "))\n"
    )

    # Get the timing stats
    wall = wall_duration(chains)
    compute = compute_duration(chains)

    # Set up string array.
    section_strings = String[]

    # Get section lines.
    if section isa Missing
        for (sec, nms) in pairs(chains.name_map)
            section_string = section_str(sec, nms)
            push!(section_strings, section_string)
        end
    else
        section in keys(chains.name_map) ||
            throw(ArgumentError("$section not found in name map."))
        section_string = section_str(section, chains.name_map[section])
        push!(section_strings, section_string)
    end

    # Return header.
    return string(
        ismissing(chains.logevidence) ? "" : "Log evidence      = $(chains.logevidence)\n",
        "Iterations        = $(range(chains))\n",
        "Number of chains  = $(size(chains, 3))\n",
        "Samples per chain = $(length(range(chains)))\n",
        ismissing(wall) ? "" : "Wall duration     = $(round(wall, digits=2)) seconds\n",
        ismissing(compute) ? "" : "Compute duration  = $(round(compute, digits=2)) seconds\n",
        section_strings...
    )
end

function indiscretesupport(
    chains::Chains,
    bounds::Tuple{Real, Real}=(0, Inf)
)
    nrows, nvars, nchains = size(chains.value)
    result = Array{Bool}(undef, nvars * (nrows > 0))
    for i in 1:nvars
        result[i] = true
        for j in 1:nrows, k in 1:nchains
            x = chains.value[j, i, k]
            if !isinteger(x) || x < bounds[1] || x > bounds[2]
                result[i] = false
                break
            end
        end
    end
    return result
end

function link(chains::Chains)
    cc = copy(chains.value.data)
    for j in axes(cc, :var)
        x = cc[:, j, :]
        # Couldn't we do this better using the transformations in `varinfo`?
        # What if the variable can take on values greater than 1, but doesn't in the sample?
        if minimum(x) > 0.0
            cc[:, j, :] = maximum(x) < 1.0 ? StatsFuns.logit.(x) : log.(x)
        end
    end
    return cc
end

### Chains specific functions ###

"""
    sort(chains::Chains[; lt = NaturalSort.natural, kwargs...])

Return a new column-sorted version of `chains`. By default, the columns are sorted in natural 
sort order. See `sort!` for a description of possible keyword arguments.
"""
function Base.sort(chains::Chains; lt = NaturalSort.natural, by=identity, kwargs...)
    val = chains.value
    var_names = dims(value, :var).val.data
    var_names = sort(var_names; by=by, lt=lt, kwargs...)
    val = val[var=var_names]

    # Sort the name map too:
    name_map = deepcopy(chains.name_map)
    for names in name_map
        sort!(names; by=by∘string, lt=lt, kwargs...)
    end

    return Chains(val[var=var_names], chains.logevidence, name_map, chains.info)
end

"""
    setinfo(chains::Chains, n::NamedTuple)

Return a new `Chains` object with a `NamedTuple` type `n` placed in the `info` field.

# Example
```julia
new_chn = setinfo(chn, NamedTuple{(:a, :b)}((1, 2)))
```
"""
function setinfo(chains::Chains, n::NamedTuple)
    return Chains(chains.value, chains.logevidence, chains.name_map, n)
end

"""
    set_section(chains::Chains, namemap)

Create a new `Chains` object from `chains` with the provided `namemap` mapping of parameter
names.

Both chains share the same underlying data. Any parameters in the chain that are unassigned
will be placed into the `:parameters` section.
"""
function set_section(chains::Chains, namemap)
    # Initialize the name map.
    _namemap = initnamemap(namemap)

    # Make sure all the names are in the new name map.
    newnames = Set(Symbol[])
    names_of_params = names(chains)
    for names in _namemap
        filter!(x -> x ∈ names_of_params, names)
        for name in names
            push!(newnames, name)
        end
    end
    missingnames = setdiff(names_of_params, newnames)

    # Assign everything that is missing to :parameters.
    if !isempty(missingnames)
        @warn "Section mapping does not contain all parameter names, " *
            "$missingnames assigned to :parameters."
        for name in missingnames
            push!(_namemap.parameters, name)
        end
    end

    return Chains(chains.value, chains.logevidence, _namemap, chains.info)
end

_default_sections(chains::Chains) = haskey(chains.name_map, :parameters) ? :parameters : nothing

function _clean_sections(chains::Chains, sections)
    return filter(sections) do section
        haskey(chains.name_map, Symbol(section))
    end
end
function _clean_sections(chains::Chains, section::Union{String,Symbol})
    return haskey(chains.name_map, Symbol(section)) ? section : ()
end
_clean_sections(::Chains, ::Nothing) = nothing


#################### Concatenation ####################

Base.cat(chains::Chains, cs::Chains...; dims = Val(1)) = _cat(dims, chains, cs...)
Base.cat(chains::T, cs::T...; dims = Val(1)) where T<:Chains = _cat(dims, chains, cs...)

Base.vcat(chains::Chains, cs::Chains...) = _cat(Val(1), chains, cs...)
Base.vcat(chains::T, cs::T...) where T<:Chains = _cat(Val(1), chains, cs...)

Base.hcat(chains::Chains, cs::Chains...) = _cat(Val(2), chains, cs...)
Base.hcat(chains::T, cs::T...) where T<:Chains = _cat(Val(2), chains, cs...)

AbstractMCMC.chainscat(chains::Chains, cs::Chains...) = _cat(Val(3), chains, cs...)

_cat(dim::Int, cs::Chains...) = _cat(Val(dim), cs...)

function _cat(::Val{1}, c1::Chains, args::Chains...)
    # check inputs
    lastiter = last(c1)
    for chains in args
        first(chains) > lastiter || throw(ArgumentError("iterations have to be sorted"))
        lastiter = last(chains)
    end
    nms = names(c1)
    all(chains -> names(chains) == nms, args) || throw(ArgumentError("chain names differ"))
    chns = chains(c1)
    all(chains -> chains(chains) == chns, args) || throw(ArgumentError("sets of chains differ"))

    # concatenate all chains
    data = mapreduce(chains -> chains.value.data, vcat, args; init = c1.value.data)
    value = DimArray(
        data,
        (
            iter = mapreduce(range, vcat, args; init=range(c1)),
            var = nms,
            chain = chns,
        )
    )

    return Chains(value, missing, c1.name_map, c1.info)
end

function _cat(::Val{2}, c1::Chains, args::Chains...)
    # check inputs
    rng = range(c1)
    all(chains -> range(chains) == rng, args) || throw(ArgumentError("chain ranges differ"))
    chns = chains(c1)
    all(chains -> chains(chains) == chns, args) || throw(ArgumentError("sets of chains differ"))

    # combine names and sections of parameters
    nms = names(c1)
    n = length(nms)
    for chains in args
        nms = union(nms, names(chains))
        n += length(names(chains))
        n == length(nms) || throw(ArgumentError("non-unique parameter names"))
    end

    name_map = mapreduce(chains -> chains.name_map, merge_union, args; init = c1.name_map)

    # concatenate all chains
    data = mapreduce(chains -> chains.value.data, hcat, args; init = c1.value.data)
    value = DimArray(data, (iter = rng, var = nms, chain = chns))

    return Chains(value, missing, name_map, c1.info)
end

function _cat(::Val{3}, c1::Chains, args::Chains...)
    # check inputs
    rng = range(c1)
    all(chains -> range(chains) == rng, args) || throw(ArgumentError("chain ranges differ"))
    nms = names(c1)
    all(chains -> names(chains) == nms, args) || throw(ArgumentError("chain names differ"))

    # concatenate all chains
    data = mapreduce(
        chains -> chains.value.data, 
        (x, y) -> cat(x, y; dims = :chain), 
        args; 
        init = c1.value.data
    )
    value = DimArray(data, (iter = rng, var = nms, chain = 1:size(data, 3)))

    # Concatenate times, if available
    starts = mapreduce(
        chains -> get(chains.info, :start_time, nothing), 
        vcat, 
        args, 
        init = get(c1.info, :start_time, nothing)
    )
    stops = mapreduce(
        chains -> get(chains.info, :stop_time, nothing), 
        vcat, 
        args, 
        init = get(c1.info, :stop_time, nothing)
    )
    nontime_props = filter(x -> !(x in [:start_time, :stop_time]), [propertynames(c1.info)...])
    new_info = NamedTuple{tuple(nontime_props...)}(tuple([c1.info[n] for n in nontime_props]...))
    new_info = merge(new_info, (start_time = starts, stop_time = stops))

    return Chains(value, missing, c1.name_map, new_info)
end

function pool_chain(chains::Chains)
    data = chains.value.data
    pool_data = reshape(permutedims(data, [1, 3, 2]), :, size(data, 2), 1)
    return Chains(pool_data, names(chains), chains.name_map; info=chains.info)
end

"""
    replacenames(chains::Chains, dict::AbstractDict)

Replace parameter names for a `Chains` object, given a dictionary of `old => new` pairs.

# Examples
```jldoctest
julia> chn = Chains(rand(100, 2, 2), ["one", "two"]);

julia> chn2 = replacenames(chn, "one" => "A");

julia> names(chn2)
2-element Vector{Symbol}:
 :A
 :two

julia> chn3 = replacenames(chn2, Dict("A" => "one", "two" => "B"));

julia> names(chn3) 
2-element Vector{Symbol}:
 :one
 :B
```
"""
replacenames(chns::Chains, dict::AbstractDict) = replacenames(chns, pairs(dict)...)
function replacenames(chns::Chains, old_new::Pair...)
    isempty(old_new) && error("you have to specify at least one replacement")

    # Set new parameter names and a new name map.
    names_of_params = copy(names(chns))
    namemap = deepcopy(chns.name_map)
    for (old, new) in old_new
        symold_symnew = Symbol(old) => Symbol(new)

        replace!(names_of_params, symold_symnew)
        for names in namemap
            replace!(names, symold_symnew)
        end
    end

    value = DimArray(
        chns.value.data,
        (iter=range(chns), var=names_of_params, chain=chains(chns))
    )

    return Chains(value, chns.logevidence, namemap, chns.info)
end


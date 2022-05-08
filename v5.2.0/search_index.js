var documenterSearchIndex = {"docs":
[{"location":"getting-started/#Getting-started","page":"Getting started","title":"Getting started","text":"","category":"section"},{"location":"getting-started/#Chains-type","page":"Getting started","title":"Chains type","text":"","category":"section"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"Chains","category":"page"},{"location":"getting-started/#MCMCChains.Chains","page":"Getting started","title":"MCMCChains.Chains","text":"Chains\n\nParameters:\n\nvalue: An AxisArray object with axes iter × var × chains\nlogevidence : A field containing the logevidence.\nname_map : A NamedTuple mapping each variable to a section.\ninfo : A NamedTuple containing miscellaneous information relevant to the chain.\n\nThe info field can be set using setinfo(c::Chains, n::NamedTuple).\n\n\n\n\n\n","category":"type"},{"location":"getting-started/#Indexing-and-parameter-Names","page":"Getting started","title":"Indexing and parameter Names","text":"","category":"section"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"Chains can be constructed with parameter names. For example, to create a chains object with","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"500 samples,\n2 parameters (named a and b)\n3 chains","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"use","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"using MCMCChains # hide\nusing Random # hide\nRandom.seed!(0) # hide\nval = rand(500, 2, 3)\nchn = Chains(val, [:a, :b])","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"By default, parameters will be given the name param_i, where i is the parameter number:","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"chn = Chains(rand(100, 2, 2))","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"We can set and get indexes for parameter 2:","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"chn_param2 = chn[1:5,2,:];","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"chn[:,2,:] = fill(4, 100, 1, 2)\nchn","category":"page"},{"location":"getting-started/#Rename-Parameters","page":"Getting started","title":"Rename Parameters","text":"","category":"section"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"Parameter names can be changed with the function replacenames:","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"replacenames","category":"page"},{"location":"getting-started/#MCMCChains.replacenames","page":"Getting started","title":"MCMCChains.replacenames","text":"replacenames(chains::Chains, dict::AbstractDict)\n\nReplace parameter names by creating a new Chains object that shares the same underlying data.\n\nExamples\n\njulia> chn = Chains(rand(100, 2, 2), [\"one\", \"two\"]);\n\njulia> chn2 = replacenames(chn, \"one\" => \"A\");\n\njulia> names(chn2)\n2-element Vector{Symbol}:\n :A\n :two\n\njulia> chn3 = replacenames(chn2, Dict(\"A\" => \"one\", \"two\" => \"B\"));\n\njulia> names(chn3) \n2-element Vector{Symbol}:\n :one\n :B\n\n\n\n\n\n","category":"function"},{"location":"getting-started/#Sections","page":"Getting started","title":"Sections","text":"","category":"section"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"Chains parameters are sorted into sections that represent groups of parameters, see  MCMCChains.group. By default, every chain contains a parameters section, to which all unassigned parameters are assigned to. Chains can be assigned a named map during construction:","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"chn = Chains(rand(100, 4, 2), [:a, :b, :c, :d])","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"The MCMCChains.set_section function returns a new Chains object:","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"chn2 = set_section(chn, Dict(:internals => [:c, :d]))","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"Note that only a and b are being shown. You can explicity retrieve an array of the summary statistics and the quantiles of the :internals section by calling describe(chn; sections = :internals), or of all variables with describe(chn; sections = nothing). Many functions such as MCMCChains.summarize support the sections keyword argument.","category":"page"},{"location":"getting-started/#Groups-of-parameters","page":"Getting started","title":"Groups of parameters","text":"","category":"section"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"You can access the names of all parameters in a chain that belong to the group name by using","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"namesingroup","category":"page"},{"location":"getting-started/#MCMCChains.namesingroup","page":"Getting started","title":"MCMCChains.namesingroup","text":"namesingroup(chains::Chains, sym::Symbol; index_type::Symbol=:bracket)\n\nReturn the parameters with the same name sym, but have a different index. Bracket indexing format in the form of :sym[index] is assumed by default. Use index_type=:dot for parameters with dot  indexing, i.e. :sym.index.\n\nIf the chain contains a parameter of name :sym it will be returned as well.\n\nExample\n\njulia> chn = Chains(rand(100, 2, 2), [\"A[1]\", \"A[2]\"]);\n\njulia> namesingroup(chn, :A)\n2-element Vector{Symbol}:\n Symbol(\"A[1]\")\n Symbol(\"A[2]\")\n\njulia> chn = Chains(rand(100, 3, 2), [\"A.1\", \"A.2\", \"B\"]);\n\njulia> namesingroup(chn, :A; index_type=:dot)\n2-element Vector{Symbol}:\n Symbol(\"A.1\")\n Symbol(\"A.2\")\n\n\n\n\n\n","category":"function"},{"location":"getting-started/#The-get-Function","page":"Getting started","title":"The get Function","text":"","category":"section"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"MCMCChains also provides a get function designed to make it easier to access parameters:","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"using MCMCChains # hide\nval = rand(6, 3, 1)\nchn = Chains(val, [:a, :b, :c]);\n\nx = get(chn, :a)","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"You can also access the variables via getproperty:","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"x.a","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"get also accepts vectors of things to retrieve, so you can call ","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"x = get(chn, [:a, :b])","category":"page"},{"location":"getting-started/#Saving-and-Loading-Chains","page":"Getting started","title":"Saving and Loading Chains","text":"","category":"section"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"Like any Julia object, a Chains object can be saved using Serialization.serialize and loaded back by Serialization.deserialize as identical as possible. Note, however, that in general this process will not work if the reading and writing are done by different versions of Julia, or an instance of Julia with a different system image. You might want to consider JLSO for saving metadata such as the Julia version and the versions of all packages installed as well.","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"using Serialization\n\nserialize(\"chain-file.jls\", chn)\nchn2 = deserialize(\"chain-file.jls\")","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"The MCMCChainsStorage.jl package also provides the ability to serialize/deserialize a chain to an HDF5 file across different versions of Julia and/or different system images.","category":"page"},{"location":"getting-started/#Exporting-Chains","page":"Getting started","title":"Exporting Chains","text":"","category":"section"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"A few utility export functions have been provided to convert Chains objects to either an Array or a DataFrame:","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"using MCMCChains # hide\n\nchn = Chains(rand(3, 2, 2), [:a, :b])\n\nArray(chn)","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"Array(chn, [:parameters])","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"By default chains are appended. This can be disabled by using the append_chains keyword  argument:","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"A = Array(chn, append_chains=false)","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"which will return a matrix for each chain. For example, for the second chain:","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"A[2]","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"Similarly, for DataFrames:","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"using DataFrames\n\nDataFrame(chn)","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"See also ?DataFrame and ?Array for more help.","category":"page"},{"location":"getting-started/#Sampling-Chains","page":"Getting started","title":"Sampling Chains","text":"","category":"section"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"MCMCChains overloads several sample methods as defined in StatsBase:","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"MCMCChains.sample(::Chains, ::Integer)","category":"page"},{"location":"getting-started/#StatsBase.sample-Tuple{Chains, Integer}","page":"Getting started","title":"StatsBase.sample","text":"sample([rng,] chn::Chains, [wv::AbstractWeights,] n; replace=true, ordered=false)\n\nSample n samples from the pooled (!) chain chn.\n\nThe keyword arguments replace and ordered determine whether sampling is performed with replacement and whether the sample is ordered, respectively. If specified, sampling probabilities are proportional to weights wv.\n\nnote: Note\nIf chn contains multiple chains, they are pooled (i.e., appended) before sampling. This ensures that even in this case exactly n samples are returned:julia> chn = Chains(randn(11, 4, 3));\n\njulia> size(sample(chn, 7)) == (7, 4, 1)\ntrue\n\n\n\n\n\n","category":"method"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"See ?sample for additional help on sampling. Alternatively, you can construct and sample from a kernel density estimator using KernelDensity.jl, see test/sampling_tests.jl.","category":"page"},{"location":"chains/#Chains","page":"Chains","title":"Chains","text":"","category":"section"},{"location":"chains/","page":"Chains","title":"Chains","text":"The methods listed below are defined in src/chains.jl.","category":"page"},{"location":"chains/","page":"Chains","title":"Chains","text":"Modules = [MCMCChains]\nPages = [\"chains.jl\"]","category":"page"},{"location":"chains/#MCMCChains.Chains-Tuple{Chains, Union{String, Symbol}}","page":"Chains","title":"MCMCChains.Chains","text":"Chains(c::Chains, section::Union{Symbol,String})\nChains(c::Chains, sections)\n\nReturn a new chain with only a specific section or multiple sections pulled out.\n\nExamples\n\njulia> chn = Chains(rand(100, 2, 1), [:a, :b], Dict(:internals => [:a]));\n\njulia> names(chn)\n2-element Vector{Symbol}:\n :a\n :b\n\njulia> chn2 = Chains(chn, :internals);\n\njulia> names(chn2)\n1-element Vector{Symbol}:\n :a\n\n\n\n\n\n","category":"method"},{"location":"chains/#Base.get-Tuple{Chains, Vector{Symbol}}","page":"Chains","title":"Base.get","text":"Base.get(c::Chains, v::Symbol; flatten=false)\nBase.get(c::Chains, vs::Vector{Symbol}; flatten=false)\n\nReturn a NamedTuple with v as the key, and matching parameter names as the values.\n\nPassing flatten=true will return a NamedTuple with keys ungrouped.\n\nExample\n\njulia> chn = Chains([1:2 3:4]);\n\njulia> get(chn, :param_1)\n(param_1 = [1; 2;;],)\n\njulia> get(chn, [:param_2])\n(param_2 = [3; 4;;],)\n\njulia> get(chn, :param_1; flatten=true)\n(param_1 = 1,)\n\n\n\n\n\n","category":"method"},{"location":"chains/#Base.get-Tuple{Chains}","page":"Chains","title":"Base.get","text":"get(c::Chains; section::Union{Symbol,AbstractVector{Symbol}}; flatten=false)\n\nReturn all parameters in a given section(s) as a NamedTuple.\n\nPassing flatten=true will return a NamedTuple with keys ungrouped.\n\nExample\n\njulia> chn = Chains([1:2 3:4], [:a, :b], Dict(:internals => [:a]));\n\njulia> get(chn; section=:parameters)\n(b = [3; 4;;],)\n\njulia> get(chn; section=[:internals])\n(a = [1; 2;;],)\n\n\n\n\n\n","category":"method"},{"location":"chains/#Base.names-Tuple{Chains, Any}","page":"Chains","title":"Base.names","text":"names(chains::Chains, sections)\n\nReturn the parameter names of the sections in the chains.\n\n\n\n\n\n","category":"method"},{"location":"chains/#Base.names-Tuple{Chains, Symbol}","page":"Chains","title":"Base.names","text":"names(chains::Chains, section::Symbol)\n\nReturn the parameter names of a section in the chains.\n\n\n\n\n\n","category":"method"},{"location":"chains/#Base.names-Tuple{Chains}","page":"Chains","title":"Base.names","text":"names(chains::Chains)\n\nReturn the parameter names in the chains.\n\n\n\n\n\n","category":"method"},{"location":"chains/#Base.range-Tuple{Chains}","page":"Chains","title":"Base.range","text":"range(chains::Chains)\n\nReturn the range of iteration indices of the chains.\n\n\n\n\n\n","category":"method"},{"location":"chains/#Base.sort-Tuple{Chains}","page":"Chains","title":"Base.sort","text":"sort(c::Chains[; lt=NaturalSort.natural])\n\nReturn a new column-sorted version of c.\n\nBy default the columns are sorted in natural sort order.\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.chains-Tuple{Chains}","page":"Chains","title":"MCMCChains.chains","text":"chains(c::Chains)\n\nReturn the names or symbols of each chain in a Chains object.\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.compute_duration-Tuple{Chains}","page":"Chains","title":"MCMCChains.compute_duration","text":"compute_duration(c::Chains; start=start_times(c), stop=stop_times(c))\n\nCalculate the compute time for all chains in seconds.\n\nThe duration is calculated as the sum of start - stop in seconds. \n\ncompute_duration is more useful in cases of parallel sampling, where wall_duration may understate how much computation time was utilitzed.\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.get_params-Tuple{Chains}","page":"Chains","title":"MCMCChains.get_params","text":"get_params(c::Chains; flatten=false)\n\nReturn all parameters packaged as a NamedTuple. Variables with a bracket in their name (as in \"P[1]\") will be grouped into the returned value as P.\n\nPassing flatten=true will return a NamedTuple with keys ungrouped.\n\nExample\n\njulia> chn = Chains(1:5);\n\njulia> x = get_params(chn);\n\njulia> x.param_1\n2-dimensional AxisArray{Int64,2,...} with axes:\n    :iter, 1:1:5\n    :chain, 1:1\nAnd data, a 5×1 Matrix{Int64}:\n 1\n 2\n 3\n 4\n 5\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.get_sections","page":"Chains","title":"MCMCChains.get_sections","text":"get_sections(chains[, sections])\n\nReturn multiple Chains objects, each containing only a single section.\n\n\n\n\n\n","category":"function"},{"location":"chains/#MCMCChains.group-Tuple{Chains, Union{String, Symbol}}","page":"Chains","title":"MCMCChains.group","text":"group(chains::Chains, name::Union{String,Symbol}; index_type::Symbol=:bracket)\n\nReturn a subset of the chain containing parameters with the same name, but a different index.\n\nBracket indexing format in the form of :name[index] is assumed by default. Use index_type=:dot for parameters with dot  indexing, i.e. :sym.index.\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.header-Tuple{Chains}","page":"Chains","title":"MCMCChains.header","text":"header(c::Chains; section=missing)\n\nReturn a string containing summary information for a Chains object. If the section keyword is used, this function prints only the relevant section header.\n\nExample\n\n# Printing the whole header.\nheader(chn)\n\n# Print only one section's header.\nheader(chn, section = :parameter)\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.max_stop-Tuple{Chains}","page":"Chains","title":"MCMCChains.max_stop","text":"max_stop(c::Chains)\n\nRetrieve the maximum of the stop times (as DateTime) from chain.info.\n\nIt is assumed that the start times are stored in chain.info.stop_time as DateTime or unix timestamps of type Float64.\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.min_start-Tuple{Chains}","page":"Chains","title":"MCMCChains.min_start","text":"min_start(c::Chains)\n\nRetrieve the minimum of the start times (as DateTime) from chain.info.\n\nIt is assumed that the start times are stored in chain.info.start_time as DateTime or unix timestamps of type Float64.\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.namesingroup-Tuple{Chains, String}","page":"Chains","title":"MCMCChains.namesingroup","text":"namesingroup(chains::Chains, sym::Symbol; index_type::Symbol=:bracket)\n\nReturn the parameters with the same name sym, but have a different index. Bracket indexing format in the form of :sym[index] is assumed by default. Use index_type=:dot for parameters with dot  indexing, i.e. :sym.index.\n\nIf the chain contains a parameter of name :sym it will be returned as well.\n\nExample\n\njulia> chn = Chains(rand(100, 2, 2), [\"A[1]\", \"A[2]\"]);\n\njulia> namesingroup(chn, :A)\n2-element Vector{Symbol}:\n Symbol(\"A[1]\")\n Symbol(\"A[2]\")\n\njulia> chn = Chains(rand(100, 3, 2), [\"A.1\", \"A.2\", \"B\"]);\n\njulia> namesingroup(chn, :A; index_type=:dot)\n2-element Vector{Symbol}:\n Symbol(\"A.1\")\n Symbol(\"A.2\")\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.replacenames-Tuple{Chains, AbstractDict}","page":"Chains","title":"MCMCChains.replacenames","text":"replacenames(chains::Chains, dict::AbstractDict)\n\nReplace parameter names by creating a new Chains object that shares the same underlying data.\n\nExamples\n\njulia> chn = Chains(rand(100, 2, 2), [\"one\", \"two\"]);\n\njulia> chn2 = replacenames(chn, \"one\" => \"A\");\n\njulia> names(chn2)\n2-element Vector{Symbol}:\n :A\n :two\n\njulia> chn3 = replacenames(chn2, Dict(\"A\" => \"one\", \"two\" => \"B\"));\n\njulia> names(chn3) \n2-element Vector{Symbol}:\n :one\n :B\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.resetrange-Tuple{Chains}","page":"Chains","title":"MCMCChains.resetrange","text":"resetrange(chains::Chains)\n\nGenerate a new chain from chains with iterations indexed by 1:n, where n is the number of samples per chain.\n\nThe new chain and chains share the same data in memory.\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.sections-Tuple{Chains}","page":"Chains","title":"MCMCChains.sections","text":"sections(c::Chains)\n\nRetrieve a list of the sections in a chain.\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.set_section-Tuple{Chains, Any}","page":"Chains","title":"MCMCChains.set_section","text":"set_section(chains::Chains, namemap)\n\nCreate a new Chains object from chains with the provided namemap mapping of parameter names.\n\nBoth chains share the same underlying data. Any parameters in the chain that are unassigned will be placed into the :parameters section.\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.setinfo-Tuple{Chains, NamedTuple}","page":"Chains","title":"MCMCChains.setinfo","text":"setinfo(c::Chains, n::NamedTuple)\n\nReturn a new Chains object with a NamedTuple type n placed in the info field.\n\nExample\n\nnew_chn = setinfo(chn, NamedTuple{(:a, :b)}((1, 2)))\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.setrange-Tuple{Chains, AbstractVector{Int64}}","page":"Chains","title":"MCMCChains.setrange","text":"setrange(chains::Chains, range::AbstractVector{Int})\n\nGenerate a new chain from chains with iterations indexed by range.\n\nThe new chain and chains share the same data in memory.\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.start_times-Tuple{Chains}","page":"Chains","title":"MCMCChains.start_times","text":"start_times(c::Chains)\n\nRetrieve the contents of c.info.start_time, or missing if no  start_time is set.\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.stop_times-Tuple{Chains}","page":"Chains","title":"MCMCChains.stop_times","text":"stop_times(c::Chains)\n\nRetrieve the contents of c.info.stop_time, or missing if no  stop_time is set.\n\n\n\n\n\n","category":"method"},{"location":"chains/#MCMCChains.wall_duration-Tuple{Chains}","page":"Chains","title":"MCMCChains.wall_duration","text":"wall_duration(c::Chains; start=min_start(c), stop=max_stop(c))\n\nCalculate the wall clock time for all chains in seconds.\n\nThe duration is calculated as stop - start, where as default stop is the latest stopping time and start is the earliest starting time.\n\n\n\n\n\n","category":"method"},{"location":"statsplots/#StatsPlots.jl","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"","category":"section"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"MCMCChains implements many functions for plotting via StatsPlots.jl.","category":"page"},{"location":"statsplots/#Simple-example","page":"StatsPlots.jl","title":"Simple example","text":"","category":"section"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"The following simple example illustrates how to use Chain to visually summarize a MCMC simulation:","category":"page"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"using MCMCChains\nusing StatsPlots\n\n# Define the experiment\nn_iter = 100\nn_name = 3\nn_chain = 2\n\n# experiment results\nval = randn(n_iter, n_name, n_chain) .+ [1, 2, 3]'\nval = hcat(val, rand(1:2, n_iter, 1, n_chain))\n\n# construct a Chains object\nchn = Chains(val, [:A, :B, :C, :D])\n\n# visualize the MCMC simulation results\nplot(chn; size=(840, 600))\n# This output is used in README.md too. # hide\nfilename = \"default_plot.svg\" # hide\nsavefig(filename); nothing # hide","category":"page"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"(Image: Default plot for Chains) \n","category":"page"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"plot(chn, colordim = :parameter; size=(840, 400))","category":"page"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"\nNote that the plot function takes the additional arguments described in the Plots.jl package.","category":"page"},{"location":"statsplots/#Mixed-density","page":"StatsPlots.jl","title":"Mixed density","text":"","category":"section"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"plot(chn, seriestype = :mixeddensity)","category":"page"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"Or, for all seriestypes, use the alternative shorthand syntax:","category":"page"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"mixeddensity(chn)","category":"page"},{"location":"statsplots/#Trace","page":"StatsPlots.jl","title":"Trace","text":"","category":"section"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"plot(chn, seriestype = :traceplot)","category":"page"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"traceplot(chn)","category":"page"},{"location":"statsplots/#Running-average","page":"StatsPlots.jl","title":"Running average","text":"","category":"section"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"meanplot(chn)","category":"page"},{"location":"statsplots/#Density","page":"StatsPlots.jl","title":"Density","text":"","category":"section"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"density(chn)","category":"page"},{"location":"statsplots/#Histogram","page":"StatsPlots.jl","title":"Histogram","text":"","category":"section"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"histogram(chn)","category":"page"},{"location":"statsplots/#Autocorrelation","page":"StatsPlots.jl","title":"Autocorrelation","text":"","category":"section"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"autocorplot(chn)","category":"page"},{"location":"statsplots/#Corner","page":"StatsPlots.jl","title":"Corner","text":"","category":"section"},{"location":"statsplots/","page":"StatsPlots.jl","title":"StatsPlots.jl","text":"corner(chn)","category":"page"},{"location":"diagnostics/#Diagnostics","page":"Diagnostics","title":"Diagnostics","text":"","category":"section"},{"location":"diagnostics/","page":"Diagnostics","title":"Diagnostics","text":"Modules = [MCMCChains]\nPages = [\n  \"discretediag.jl\",\n  \"gelmandiag.jl\",\n  \"gewekediag.jl\",\n  \"heideldiag.jl\",\n  \"rafterydiag.jl\",\n  \"rstar.jl\",\n  \"ess.jl\"\n]","category":"page"},{"location":"diagnostics/#MCMCDiagnosticTools.discretediag-Tuple{Chains{var\"#s12\", A} where {var\"#s12\"<:Real, A<:(AxisArrays.AxisArray{var\"#s12\", 3})}}","page":"Diagnostics","title":"MCMCDiagnosticTools.discretediag","text":"discretediag(chains::Chains{<:Real}; sections, kwargs...)\n\nDiscrete diagnostic where method can be [:weiss, :hangartner, :DARBOOT, MCBOOT, :billinsgley, :billingsleyBOOT].\n\n\n\n\n\n","category":"method"},{"location":"diagnostics/#MCMCDiagnosticTools.rstar-Tuple{MLJModelInterface.Supervised, Chains}","page":"Diagnostics","title":"MCMCDiagnosticTools.rstar","text":"rstar(rng=Random.GLOBAL_RNG, classifier, chains::Chains; kwargs...)\n\nCompute the R^* convergence diagnostic of the MCMC chains with the classifier.\n\nThe keyword arguments supported here are the same as those in rstar for arrays of samples and chain indices.\n\nExamples\n\njulia> using MLJBase, MLJDecisionTreeInterface, Statistics\n\njulia> chains = Chains(fill(4.0, 100, 2, 3));\n\nOne can compute the distribution of the R^* statistic with the probabilistic classifier.\n\njulia> distribution = rstar(DecisionTreeClassifier(), chains);\n\njulia> isapprox(mean(distribution), 1; atol=0.1)\ntrue\n\nFor deterministic classifiers, a single R^* statistic is returned.\n\njulia> decisiontree_deterministic = Pipeline(\n           DecisionTreeClassifier();\n           operation=predict_mode,\n       );\n\njulia> value = rstar(decisiontree_deterministic, chains);\n\njulia> isapprox(value, 1; atol=0.2)\ntrue\n\n\n\n\n\n","category":"method"},{"location":"diagnostics/#MCMCDiagnosticTools.ess_rhat-Tuple{Chains}","page":"Diagnostics","title":"MCMCDiagnosticTools.ess_rhat","text":"ess_rhat(chains::Chains; duration=compute_duration, kwargs...)\n\nEstimate the effective sample size and the potential scale reduction.\n\nESS per second options include duration=MCMCChains.compute_duration (the default) and duration=MCMCChains.wall_duration.\n\n\n\n\n\n","category":"method"},{"location":"modelstats/#Model-selection","page":"Model selection","title":"Model selection","text":"","category":"section"},{"location":"modelstats/","page":"Model selection","title":"Model selection","text":"The methods listed below are defined in src/modelstats.jl.","category":"page"},{"location":"modelstats/","page":"Model selection","title":"Model selection","text":"Modules = [MCMCChains]\nPages = [\"modelstats.jl\"]","category":"page"},{"location":"modelstats/#MCMCChains.dic-Tuple{Chains, Function}","page":"Model selection","title":"MCMCChains.dic","text":"dic(chain::Chains, logpdf::Function) -> (DIC, pD)\n\nCompute the deviance information criterion (DIC). (Smaller is better)\n\nNote: DIC assumes that the posterior distribution is approx. multivariate Gaussian and tends to select overfitted models.\n\nReturns:\n\nDIC: The calculated deviance information criterion\npD: The effective number of parameters\n\nUsage:\n\nchn ... # sampling results\nlpfun = function f(chain::Chains) # function to compute the logpdf values\n    niter, nparams, nchains = size(chain)\n    lp = zeros(niter + nchains) # resulting logpdf values\n    for i = 1:nparams\n        lp += map(p -> logpdf( ... , x), Array(chain[:,i,:]))\n    end\n    return lp\nend\n\nDIC, pD = dic(chn, lpfun)\n\n\n\n\n\n","category":"method"},{"location":"stats/#Posterior-statistics","page":"Posterior statistics","title":"Posterior statistics","text":"","category":"section"},{"location":"stats/","page":"Posterior statistics","title":"Posterior statistics","text":"The methods listed below are defined in src/stats.jl.","category":"page"},{"location":"stats/","page":"Posterior statistics","title":"Posterior statistics","text":"autocor\ndescribe\nmean\nsummarystats\nquantile","category":"page"},{"location":"stats/#StatsBase.autocor","page":"Posterior statistics","title":"StatsBase.autocor","text":"autocor(\n    chains;\n    append_chains = true,\n    demean = true,\n    [lags,]\n    kwargs...,\n)\n\nCompute the autocorrelation of each parameter for the chain.\n\nThe default lags are [1, 5, 10, 50], upper-bounded by n - 1 where n is the number of samples used in the estimation.\n\nSetting append_chains=false will return a vector of dataframes containing the autocorrelations for each chain.\n\n\n\n\n\n","category":"function"},{"location":"stats/#DataAPI.describe","page":"Posterior statistics","title":"DataAPI.describe","text":"describe(io, chains[;\n         q = [0.025, 0.25, 0.5, 0.75, 0.975],\n         etype = :bm,\n         kwargs...])\n\nPrint the summary statistics and quantiles for the chain.\n\n\n\n\n\n","category":"function"},{"location":"stats/#Statistics.mean","page":"Posterior statistics","title":"Statistics.mean","text":"mean(chains[, params; kwargs...])\n\nCalculate the mean of a chain.\n\n\n\n\n\n","category":"function"},{"location":"stats/#StatsBase.summarystats","page":"Posterior statistics","title":"StatsBase.summarystats","text":"function summarystats(\n    chains;\n    sections = _default_sections(chains),\n    append_chains= true,\n    method::AbstractESSMethod = ESSMethod(),\n    maxlag = 250,\n    etype = :bm,\n    kwargs...\n)\n\nCompute the mean, standard deviation, naive standard error, Monte Carlo standard error, and effective sample size for each parameter in the chain.\n\nSetting append_chains=false will return a vector of dataframes containing the summary statistics for each chain.\n\nWhen estimating the effective sample size, autocorrelations are computed for at most maxlag lags.\n\n\n\n\n\n","category":"function"},{"location":"stats/#Statistics.quantile","page":"Posterior statistics","title":"Statistics.quantile","text":"quantile(chains[; q = [0.025, 0.25, 0.5, 0.75, 0.975], append_chains = true, kwargs...])\n\nCompute the quantiles for each parameter in the chain.\n\nSetting append_chains=false will return a vector of dataframes containing the quantiles for each chain.\n\n\n\n\n\n","category":"function"},{"location":"makie/#Makie.jl-plots","page":"Makie.jl","title":"Makie.jl plots","text":"","category":"section"},{"location":"makie/","page":"Makie.jl","title":"Makie.jl","text":"This page shows an example of plotting MCMCChains.jl with Makie.jl. The example is meant to provide an useful basis to build upon. Let's define some random chain and load the required packages:","category":"page"},{"location":"makie/","page":"Makie.jl","title":"Makie.jl","text":"using MCMCChains\n\nchns = Chains(randn(300, 5, 3), [:A, :B, :C, :D, :E])","category":"page"},{"location":"makie/","page":"Makie.jl","title":"Makie.jl","text":"A basic way to visualize the chains is to show the drawn samples at each iteration. Colors depict different chains.","category":"page"},{"location":"makie/","page":"Makie.jl","title":"Makie.jl","text":"using CairoMakie\nCairoMakie.activate!(; type=\"svg\")\n\nparams = names(chns, :parameters)\n\nn_chains = length(chains(chns))\nn_samples = length(chns)\n\nfig = Figure(; resolution=(1_000, 800))\n\nfor (i, param) in enumerate(params)\n    ax = Axis(fig[i, 1]; ylabel=string(param))\n    for chain in 1:n_chains\n        values = chns[:, param, chain]\n        lines!(ax, 1:n_samples, values; label=string(chain))\n    end\n\n    hideydecorations!(ax; label=false)\n    if i < length(params)\n        hidexdecorations!(ax; grid=false)\n    else\n        ax.xlabel = \"Iteration\"\n    end\nend\n\nfig","category":"page"},{"location":"makie/","page":"Makie.jl","title":"Makie.jl","text":"Next, we can add a second row of plots next to it which show the density estimate for these samples:","category":"page"},{"location":"makie/","page":"Makie.jl","title":"Makie.jl","text":"for (i, param) in enumerate(params)\n    ax = Axis(fig[i, 2]; ylabel=string(param))\n    for chain in 1:n_chains\n        values = chns[:, param, chain]\n        density!(ax, values; label=string(chain))\n    end\n\n    hideydecorations!(ax)\n    if i < length(params)\n        hidexdecorations!(ax; grid=false)\n    else\n        ax.xlabel = \"Parameter estimate\"\n    end\nend\n\naxes = [only(contents(fig[i, 2])) for i in 1:length(params)]\nlinkxaxes!(axes...)\n\nfig","category":"page"},{"location":"makie/","page":"Makie.jl","title":"Makie.jl","text":"Finally, let's add a simple legend. Thanks to setting label above, this legend will have the right labels:","category":"page"},{"location":"makie/","page":"Makie.jl","title":"Makie.jl","text":"axislegend(first(axes))\n\nfig","category":"page"},{"location":"gadfly/#Gadfly.jl-plots","page":"Gadfly.jl","title":"Gadfly.jl plots","text":"","category":"section"},{"location":"gadfly/","page":"Gadfly.jl","title":"Gadfly.jl","text":"To plot the Chains via Gadfly.jl, use the DataFrames constructor:","category":"page"},{"location":"gadfly/","page":"Gadfly.jl","title":"Gadfly.jl","text":"using DataFrames\nusing CategoricalArrays\nusing Gadfly\nwrite_svg(path, p; w=6inch, h=4inch) = Gadfly.draw(Gadfly.SVG(path, w, h), p) # hide\nusing MCMCChains\n\nchn = Chains(randn(100, 2, 3), [:A, :B])\ndf = DataFrame(chn)\ndf[!, :chain] = categorical(df.chain)\n\nplot(df, x=:A, color=:chain, Geom.density, Guide.ylabel(\"Density\"))","category":"page"},{"location":"gadfly/#Multiple-parameters","page":"Gadfly.jl","title":"Multiple parameters","text":"","category":"section"},{"location":"gadfly/","page":"Gadfly.jl","title":"Gadfly.jl","text":"Or, to show multiple parameters in one plot, use DataFrames.stack","category":"page"},{"location":"gadfly/","page":"Gadfly.jl","title":"Gadfly.jl","text":"sdf = stack(df, names(chn), variable_name=:parameter)\nfirst(sdf, 5)","category":"page"},{"location":"gadfly/","page":"Gadfly.jl","title":"Gadfly.jl","text":"and Gadfly.Geom.subplot_grid","category":"page"},{"location":"gadfly/","page":"Gadfly.jl","title":"Gadfly.jl","text":"plot(sdf, ygroup=:parameter, x=:value, color=:chain,\n    Geom.subplot_grid(Geom.density), Guide.ylabel(\"Density\"))","category":"page"},{"location":"gadfly/","page":"Gadfly.jl","title":"Gadfly.jl","text":"This is very flexible. For example, we can look at the first two chains only by using DataFrames.filter","category":"page"},{"location":"gadfly/","page":"Gadfly.jl","title":"Gadfly.jl","text":"first_chain = filter([:chain] => c -> c == 1 || c == 2, sdf)\n\nplot(first_chain, xgroup=:parameter, ygroup=:chain, x=:value,\n    Geom.subplot_grid(Geom.density, Guide.xlabel(orientation=:horizontal)),\n    Guide.xlabel(\"Parameter\"), Guide.ylabel(\"Chain\"))","category":"page"},{"location":"gadfly/#Trace","page":"Gadfly.jl","title":"Trace","text":"","category":"section"},{"location":"gadfly/","page":"Gadfly.jl","title":"Gadfly.jl","text":"plot(first_chain, ygroup=:parameter, x=:iteration, y=:value, color=:chain,\n    Geom.subplot_grid(Geom.point), Guide.ylabel(\"Sample value\"))","category":"page"},{"location":"#MCMCChains","page":"MCMCChains","title":"MCMCChains","text":"","category":"section"},{"location":"","page":"MCMCChains","title":"MCMCChains","text":"Implementation of Julia types for summarizing MCMC simulations and utility functions for diagnostics and visualizations.","category":"page"},{"location":"summarize/#Summarize","page":"Summarize","title":"Summarize","text":"","category":"section"},{"location":"summarize/","page":"Summarize","title":"Summarize","text":"The methods listed below are defined in src/summarize.jl.","category":"page"},{"location":"summarize/","page":"Summarize","title":"Summarize","text":"Modules = [MCMCChains]\nPages = [\"summarize.jl\"]","category":"page"},{"location":"summarize/#MCMCChains.summarize-Tuple{Chains, Vararg{Any}}","page":"Summarize","title":"MCMCChains.summarize","text":"summarize(chains, funs...[; sections, func_names = []])\n\nSummarize chains in a ChainsDataFrame.\n\nExamples\n\nsummarize(chns) : Complete chain summary\nsummarize(chns[[:parm1, :parm2]]) : Chain summary of selected parameters\nsummarize(chns; sections=[:parameters])  : Chain summary of :parameters section\nsummarize(chns; sections=[:parameters, :internals]) : Chain summary for multiple sections\n\n\n\n\n\n","category":"method"}]
}

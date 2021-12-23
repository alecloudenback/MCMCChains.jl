
@testset "Basic chain behavior" begin
    
    chn = Chains(rand(100, 2, 1), [:a, :b], Dict(:internals => [:a]));
    names(chn) == [:a, :b]

    chn = Chains(chn, :internals)
    names(chn) == [:a]
end
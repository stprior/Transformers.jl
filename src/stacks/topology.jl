#=
examples
x => b => c  ==> b = m[1](x) ; c = m[2](b)
x => 3 ==> x => a => a => a ==> x = m[1](a); a = m[1](a); a = m[1](a)
(x, m) => a => b => c ==> a = m[1](x , m); b = m[2](b); c = m[3](b)
((x, m) => x) => 3 ==> (x = m[1](x, m)); (x = m[2](x, m)); (x = m[3](x, m))
(((x, m) => x:(x, m)) => 3) ==> (x = m[1](x,m)); (x = m[2](x,m)) ;(x = m[3](x,m))
=#
include("./new_topo.jl")

"""
    nntopo"pattern"

the @nntopo string
"""
macro nntopo_str(str)
    NNTopo(str)
end

"""
    @nntopo pattern

create the function according to the given pattern
"""
macro nntopo(expr)
    NNTopo(interpolate(__module__, expr))
end


isinterpolate(x) = false
isinterpolate(ex::Expr) = ex.head == :($)
interpolate(m::Module, x) = x
function interpolate(m::Module, ex::Expr)
    if isinterpolate(ex)
        return @eval(m, $(ex.args[1]))
    else
        for (i, e) ∈ enumerate(ex.args)
            ex.args[i] = interpolate(m, e)
        end
    end
    ex
end

"""
    NNTopo(s)

the type of a sequence of function
"""
struct NNTopo{FS} end

Base.getproperty(nt::NNTopo, s::Symbol) = s == :fs ? Base.getproperty(nt, Val(:fs)) : error("type NNTopo has no field $s")
Base.getproperty(::NNTopo{FS}, ::Val{:fs}) where FS = string(FS)

NNTopo(s::String) = NNTopo(Meta.parse(s))
NNTopo(ex::Expr) = islegal(ex) && !hascollect(getin(leftmost(ex))) ?
  NNTopo{Symbol(ex)}() :
  error("topo pattern illegal")

genline(name, arg::Symbol, m, i::Int) = Expr(:(=), name, Expr(:call, :($m[$i]), arg))
genline(name, args::Expr, m, i::Int) = Expr(:(=), name, Expr(:call, :($m[$i]), args.args...))

nntopo_impl(s::Symbol) = nntopo_impl(string(s))
nntopo_impl(sf::String) = nntopo_impl(Meta.parse(sf))
function nntopo_impl(pattern)
    m = :model
    xs = :xs

    code = to_code(pattern)

    if istuple(code.in)
        pref = Expr(:(=), code.in, xs)
    else
        pref = Expr(:(=), Expr(:tuple, code.in), xs)
    end

    fbody = Any[:block]
    push!(fbody, pref)
    for (i, l) ∈ enumerate(code.lines)
        push!(fbody, genline(l..., m, i))
    end

    push!(fbody, code.out)

    Expr(fbody...)
end

@generated function (nt::NNTopo{FS})(model, xs...) where {FS}
    return nntopo_impl(FS)
end

function Base.show(io::IO, nt::NNTopo)
    println(io, "NNTopo{\"$(nt.fs)\"}")
    print_topo(io, nt)
    io
end

print_topo(nt::NNTopo; models=nothing) = print_topo(stdout, nt; models=models)
function print_topo(io::IO, nt::NNTopo; models=nothing)
    code = to_code(Meta.parse(nt.fs))
    farg = istuple(code.in) ? join(code.in.args, ", ") : string(code.in)
    println(io, "topo_func(model, $farg)")
    for (i, l) ∈ enumerate(code.lines)
        name = string(l[1])
        args = istuple(l[2]) ? string(l[2]) : "($(l[2]))"
        model = models === nothing ? "model[$i]" : string(models[i])
        println(io, "\t$name = $model$args")
    end
    println(io, "\t$(code.out)")
    println("end")
end


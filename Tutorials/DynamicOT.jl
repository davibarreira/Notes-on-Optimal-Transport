### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 3e1a0dc2-6d70-4b9a-afb5-fdee4983a2a8
using PlutoUI, OptimalTransport, VegaLite, LinearAlgebra, Distributions, Distances

# ╔═╡ 8eec5c9e-bef6-11eb-282c-297b67ffdd26
md"""
# Optimal Transport Dynamic Formulation

In "Optimal Transport - Old and New", Villani points out that a dynamic description of Optimal Tranport, i.e. adding a **time** component, might have benefits, such as a more complete description of the transport and a richer mathematical structure.
In the canonical OT problem, one specifyies only the final cost of transportation, whithout caring (or modelling) for the underlying space in which the measures live. All this information would be already imbued in the cost function. One can instead have a manifold, in which the cost between two point will be given by the geodesic distance. Thus, one can derive the cost function from such more fundamental characteristics.
"""

# ╔═╡ b9e7b154-9fa1-4821-bc45-f4fc334ca045
md"""

Let's start with some definitions.

**Def**: Let $(X,d)$ be a metric space. Then, a curve on $(X,d)$ is a continuous functions $\omega:[a,b] \to X$ (e.g. $[a,b] = [0,1]$).

**Def (Metric Derivative)**: For a curve $\omega:[0,1] \to X$, the metric derivative
of $\omega$ at time $t$, denoted by $|\omega'|(t)$, is
```math
|\omega'|(t) := \lim_{h \to 0} \frac{d(\omega(t+h),\omega(t)}{|h|}
```

**Def (Absolute Continuity)**: A curve $\omega:[0,1] \to X$ is said to be
absolute continuous if there exists a $g \in L^1([0,1])$ such that
```math
d(\omega(t_0),\omega(t_1)) \leq \int^{t_1}_{t_0}g(s) ds
```
for every $t_0 < t_1$.
The set of all absolute continuous curves defined on $[0,1]$ and valued on
$X$ is denoted by $\text{AC}(X)$.

**Def**: For a curve $\omega:[0,1] \to X$, we define
```math
\text{Length}(\omega) := \sup \left\{
	\sum^{n-1}_{k=0} d(\omega(t_k),\omega(t_k+1)): n\geq 1,
	0 = t_0 < t_1 < ... < t_n = 1
\right\}.
```

**Proposition**: For any curve $\omega \in \text{AC}(X)$, we have
```math
\text{Length}(\omega) = \int^1_0 |\omega'|(t) dt.
```

"""


# ╔═╡ 7f7b005b-16a9-4c81-b154-5d4274d1d61c
PlutoUI.LocalResource("./lengths.svg")

# ╔═╡ 78ec8d37-6e18-47c7-9b3b-d2355a060eb0
md"""

**Def (Geodesic)**: A curve $\omega:[0,1] \to X$ is said to be a geodesic between
$x\in X$ and $y\in X$ if it minimizes the length among all curves such that
$\omega(0) = x$ and $\omega(1) = y$.

**Def (Length Sapce)**: A space $(X,d)$ is said to be a *length space* if it holds that
```math
d(x,y) =\inf \{ \text{Length}(\omega) : \omega \in \text{AC}(X), \omega(0) = x,\omega(1) = 1 \}.
```

**Def (Geodesic Sapce)**: A space $(X,d)$ is said to be a *geodesic space* if
it holds that
```math
d(x,y) =\min \{ \text{Length}(\omega) : \omega \in \text{AC}(X), \omega(0) = x,\omega(1) = 1 \},
```
i.e., it is a length space with geodesics between all points.

**Def (Constant-Speed Geodesics)**: In a length space, a curve
$\omega:[0,1] \to X$ is said to be a constant-speed geodesic between $\omega(0)$
and $\omega(1)$ if it satisfies
```math
d(\omega(t),\omega(s) ) = |t-s| d(\omega(0),\omega(1)), \quad
\forall t,s \in [0,1].
```

We can prove that every curve with this property is also a geodesic. 

"""


# ╔═╡ bf2d96ce-3a28-4b35-8c89-5d17ff32ffd3
md"""
---
#### Displacement Interpolation
Using the idea of geodesic in Wasserstein space, one can define the notion of
displacement interpolation. Note in the left the "Euclidean interpolation", which
is given by $\mu t + \nu (1-t)$, while in the right we have the displacement interpolation.

---
"""

# ╔═╡ 685fa3c8-c3bb-42aa-ad79-d73b7dd4a292
@bind t Slider(0:0.1:1)

# ╔═╡ 90cb48e4-374e-4369-ba04-9acbd7e25b1f


# ╔═╡ 458e0b3a-8277-4616-b19f-d197beb3ad66
begin
	μ = Normal(0,2)
	ν = Normal(30,3)
end

# ╔═╡ 753a6bc3-bdaf-4807-bef0-18171e4f2910
function otplan(μ, ν)
    # Use T instead of γ to indicate that this is a Monge map.
    T(x) = quantile(ν, cdf(μ, x))
    return T
end

# ╔═╡ 77b8cc32-92c7-4822-831c-c2ce17ead071
"""
`plotDistribution(μ::distribution,ν::distribution,x=collect(-10:0.1:10))`
'x' is the plotting domain.

This function returns both the pdfs and cdfs of the pair
of probability measures.
"""
function plotDistributions( μ::Distributions.UnivariateDistribution,
                            ν::Distributions.UnivariateDistribution,
							colormu="blue",colornu="red",
							scalemu=1.0, scalenu=1.0,
                            x=collect(-10:0.1:10))
    
    x  = collect(-10:0.1:40)
    y  = pdf(μ,x)*scalemu
    z  = pdf(ν,x)*scalenu
    cy = cdf(μ,x)
    cz = cdf(ν,x);
    cmin = min.(cy,cz)
    cmax = max.(cy,cz)

    pdf1 = @vlplot(:line,x={x,type="quantitative"},
			y={y,type="quantitative",scale={domain=[0,0.25]}},color={value=colormu})
    pdf2 = @vlplot(:line,x={x,type="quantitative"},
			y={z,type="quantitative",scale={domain=[0,0.25]}},color={value=colornu})
    pdfs = @vlplot()+ pdf1 + pdf2

    return pdfs
end


# ╔═╡ 0d6c295b-3dc9-45b6-ab2a-d12b0cff9eb2
p1= plotDistributions(μ,ν,"blue","red",1,1);

# ╔═╡ 30512da5-2a0b-439e-a70d-32d38d7b928b
p2= plotDistributions(μ,ν,"green","green",1-t,t);

# ╔═╡ a728d84e-a3de-4234-a28a-1456f949d075
function plotInterpolation( μ,ν,Tp,colormu="blue",colornu="red",scale=1.0)
	x  = collect(-10:0.1:10)
    y  = pdf(μ,x)
    z  = pdf(ν,Tp(x))
	w  = y*scale + z*(1.0-scale)
	xt = x*scale + Tp(x)*(1-scale)
   

    pdf1 = @vlplot(:line,x={x,type="quantitative",scale={domain=[-10,40]}},
			y={y,type="quantitative",scale={domain=[0,0.25]}},color={value=colormu})
    pdf2 = @vlplot(:line,x={Tp(x),type="quantitative"},
			y={z,type="quantitative",scale={domain=[0,0.25]}},color={value=colornu})
	pdf3 = @vlplot(:line,x={xt,type="quantitative",scale={domain=[-10,40]}},
			y={w,type="quantitative",scale={domain=[0,0.25]}},color={value="green"})
    pdfs = @vlplot()+ pdf3+ pdf1 + pdf2

    return pdfs
    
#     xs  = collect(-10:0.1:40)
#     z  = pdf(μ,xs)
# 	w  = pdf(ν,xs)
# 	# y  = z*scale+w*(1-scale)
# 	xt = xs*scale + Tp.(xs)*(1-scale)

#     pdf1 = @vlplot(:line,x={xt,type="quantitative",scale={domain=[-10,40]}},
# 			y={w,type="quantitative",scale={domain=[0,0.25]}},color={value=colormu})

    return pdf1
end


# ╔═╡ 64862f33-db02-445c-b83a-8e4b9c35d9d0
p3 = plotInterpolation(μ,ν,otplan(μ,ν),"blue","red",1-t);

# ╔═╡ 04c8df6b-c8d2-47a3-965c-25d7bbb7e44f
[p2+p1 p3]

# ╔═╡ Cell order:
# ╟─8eec5c9e-bef6-11eb-282c-297b67ffdd26
# ╟─b9e7b154-9fa1-4821-bc45-f4fc334ca045
# ╟─7f7b005b-16a9-4c81-b154-5d4274d1d61c
# ╟─78ec8d37-6e18-47c7-9b3b-d2355a060eb0
# ╟─bf2d96ce-3a28-4b35-8c89-5d17ff32ffd3
# ╟─685fa3c8-c3bb-42aa-ad79-d73b7dd4a292
# ╟─04c8df6b-c8d2-47a3-965c-25d7bbb7e44f
# ╟─90cb48e4-374e-4369-ba04-9acbd7e25b1f
# ╟─458e0b3a-8277-4616-b19f-d197beb3ad66
# ╟─0d6c295b-3dc9-45b6-ab2a-d12b0cff9eb2
# ╟─30512da5-2a0b-439e-a70d-32d38d7b928b
# ╟─64862f33-db02-445c-b83a-8e4b9c35d9d0
# ╟─753a6bc3-bdaf-4807-bef0-18171e4f2910
# ╟─77b8cc32-92c7-4822-831c-c2ce17ead071
# ╟─3e1a0dc2-6d70-4b9a-afb5-fdee4983a2a8
# ╟─a728d84e-a3de-4234-a28a-1456f949d075

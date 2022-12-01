@time begin println("\n$(prod(['#' for _ in 1:50]))\n\nImporting dependencies..")
using LinearAlgebra
using Statistics
using DataFrames
using CSV
include("julia_fn.jl")
end;

@time begin println("\nImporting, formatting and scaling data..")
load_trn = CSV.read("TrainingData/neodata/soltani_14d_nonoise_1200.csv",DataFrame)
load_vld = CSV.read("ValidationData/neodata/soltani_14d_nonoise_1200.csv",DataFrame)
load_tst = CSV.read("TestData/neodata/soltani_14d_nonoise_100.csv",DataFrame)

y_trn = load_trn.target
X_trn = load_trn[:,1:14]

y_vld = load_vld.target
X_vld = load_vld[:,1:14]

y_tst = load_tst.target
X_tst = load_tst[:,1:14]

features = names(X_trn)

(m,s) = ScalarFit(X_trn)

X_trn = ScalarUse(X_trn,m,s)
X_vld = ScalarUse(X_vld,m,s)
X_tst = ScalarUse(X_tst,m,s);
end;

@time remove,cond_numbers = feature_conditioner(X_trn);

function feature_conditioner(X :: DataFrame)
    removal :: Vector{String} = []
    conditioning :: Vector{Float64} = [cond(Matrix(X))]
    for _ in 1:(length(names(X))-1)
        conds :: Dict{String,Float64} = Dict()
        for feature in names(X)
            if !(feature in removal)
                temp = copy(removal)
                push!(temp,feature)
                conds[feature] = cond(Matrix(select(X,Not(temp))))
            end
        end
        push!(removal,findmin(conds)[2])
        append!(conditioning,cond(Matrix(select(X,Not(removal)))))
    end
    return (removal,conditioning)
end

# More research should be done for feature extraction methods!
# https://medium.com/analytics-vidhya/feature-selection-extended-overview-b58f1d524c1c

begin
function fisher_feature(X :: DataFrame , y)
    
    num_classes = length(unique(y))
    num_feature = length(names(X))

    vars = zeros(num_classes,num_feature)
    means = zeros(num_classes,num_feature)

    for (ci,c) ∈ enumerate(unique(y))
        for (fi,f) ∈ enumerate(names(X))
            vars[ci,fi] = var(X[:,f][y .== c,:])
            means[ci,fi] = mean(X[:,f][y .== c,:])
        end
    end

    return means,vars

    mean_between = zeros(num_feature)
    for f ∈ 1:num_feature
        mean_between[f] = sum(abs.(means[:,f]))         
    end

    vars_between = zeros(num_feature)
    for f ∈ 1:num_feature
        vars_between[f] = sum(vars[:,f])         
    end

    return mean_between,vars_between

end

means,vars = fisher_feature(X_trn,y_trn)

for i in 1:14
println("$(names(X_trn)[i])\t: $((means./vars)[i]*100)")
end

end

round.((10*means./vars).^2,digits=5)

(means\vars)*ones(14)

names(X_trn)
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

@time model = qda_fit(X_trn,y_trn);
@time qda_score(X_trn,y_trn,model)
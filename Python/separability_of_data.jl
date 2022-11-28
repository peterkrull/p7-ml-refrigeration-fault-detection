@time begin println("Importing dependencies")
using HypothesisTests
using TimerOutputs
using DataFrames
using Statistics
using ImageView
using CSV
end

## Function definitions
@time begin println("Importing custom functions")
include("julia_fn.jl")
end

# Load data 
@time begin println("Loading data")
trn_data = CSV.read("TrainingData/neodata/fault_all_2000.csv",DataFrame)
vld_data = CSV.read("ValidationData/neodata/soltani_14d_nonoise_1200.csv",DataFrame)
tst_data = CSV.read("TestData/neodata/soltani_14d_nonoise_100.csv",DataFrame);
end

# Split data and targets
@time begin println("Split data and targets")
y_trn = trn_data.target
X_trn = trn_data[:,1:14]

y_vld = vld_data.target
X_vld = vld_data[:,1:14]

y_tst = tst_data.target
X_tst = tst_data[:,1:14];
end

# Compute mean and variance for scaling
@time begin println("Apply standard scaling to data")
m,s = ScalarFit(X_trn);

X_trn = ScalarUse(X_trn,m,s);
X_vld = ScalarUse(X_vld,m,s);
X_tst = ScalarUse(X_tst,m,s);
end

# matrix = zeros(length(unique(y_trn)),length(unique(y_trn)))

# @time for (i,ci) in enumerate(unique(y_trn)) , (j,cj) in enumerate(unique(y_trn)) 
#     println("Comparing class $ci and $cj")
#     matrix[i,j] = SeparabilityIndex(X_trn[(y_trn .== ci) .| (y_trn .== cj),:],y_trn[(y_trn .== ci) .| (y_trn .== cj),:])
# end

@time ICD(X_tst);
@time BCD(X_tst,X_tst);

class_1 = 2
class_2 = 0

@time SeparabilityIndex(X_tst[(y_tst .== class_1) .| (y_tst .== class_2),:],y_tst[(y_tst .== class_1) .| (y_tst .== class_2),:])

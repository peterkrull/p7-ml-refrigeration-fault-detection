using TimerOutputs

function ScalarFit(X)
    X = Matrix(X)
    N,f = size(X)

    means = [mean(X[:,i]) for i in 1:f]
    stds = [std(X[:,i]) for i in 1:f]

    return means,stds
end

function ScalarUse(X,m,s)
    X = Matrix(X)
    N,f = size(X)

    X_out = zeros((N,f))
    for (i,row) in enumerate(eachrow(X))
        X_out[i,:] = (row-m)./s
    end
    return X_out
end

function ScalarUse(X::DataFrame,m,s)
    return DataFrame(ScalarUse(Matrix(X),m,s),names(X))
end


function ICD(X) :: Vector{Float64}
    X :: Matrix{Float64} = Matrix(X)
    N :: UInt32,_ = size(X)
    return [X[i,:]'*X[j,:] for i in 1:(N-1) for j in (i+1):N]
end

function BCD(A,B) :: Vector{Float64}
    A :: Matrix{Float64} = Matrix(A)
    B :: Matrix{Float64} = Matrix(B)
    return [A[a,:]'*B[b,:] for a in 1:size(A)[1] for b in 1:size(B)[1]]
end

function SeparabilityIndex(X,y)
    X = Matrix(X)
    classes = unique(y)
    vec = zeros(length(classes))
    for (i,c) in enumerate(classes)
        pos = X[y .== c , :]
        neg = X[y .!= c , :]
        
        dist_pos = ICD(pos)
        dist_neg = BCD(pos,neg)
        
        vec[i] =  ApproximateTwoSampleKSTest(dist_pos,dist_neg).δn
    end
    return sum(vec)/length(classes)
end

euclid_dist(a,b) = a'*b


## QDA ##

function qda_fit(X,y)
    M = [mean(Matrix(X[y .== c , :]), dims = 1)' for c in unique(y)];
    P = [length(y[y .== c , :])/length(y) for c in unique(y)];
    S = [cov(Matrix(X[y .== c , :])) for c in unique(y)];
    return (M,P,S) # return model
end

function qda_discriminant_function(X,sinv,m,consts)    
    return consts -0.5*dot((X-m)',sinv*(X-m)) 
end

function qda_classifier(X,model)
    (M,P,S) = model
    Sinv = [pinv(s) for s in S]
    Consts = [log(p)-real(0.5*log(Complex(det(s)))) for (s,p) in zip(S,P)]

    estimates :: Vector{Int} = zeros(size(X)[1])
    probabilities :: Vector{Float64} = zeros(size(M)[1])
    for (i,sample) in enumerate(eachrow(Matrix(X)))
        for (c,(m,sinv,consts)) in enumerate(zip(M,Sinv,Consts))
            probabilities[c] = qda_discriminant_function(sample,sinv,m,consts)
        end
        estimates[i] = argmax(probabilities)
    end
    return estimates
end

function qda_score(X,y,model)
    y_hat= unique(y)[qda_classifier(X,model)]
    return sum(y_hat .== y)/length(y)
end
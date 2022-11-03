%% Data extraction


data_trn = readmatrix('C:\Users\laula\OneDrive\Documents\GitHub\p7-ml-refrigeration-fault-detection\TrainingData\neodata\fault_all_noise_67.csv');
data_val = readmatrix('C:\Users\laula\OneDrive\Documents\GitHub\p7-ml-refrigeration-fault-detection\ValidationData\neodata\fault_all_noise_67.csv');

data_trn = datasample(data_trn,6000);
data_val = datasample(data_val,500);

x_trn = data_trn(:,1:11);
y_trn = data_trn(:,12);

x_val = data_val(:,1:11);
y_val = data_val(:,12);

[x_trn, C, S] = normalize(x_trn);
x_val = normalize(x_val,'center',C,'scale',S);


clear  data_val
%% Training
disp("Start training")
t = templateSVM('KernelFunction','rbf','KernelScale', 1/(2*0.01),'BoxConstraint',1000);
Mdl = fitcecoc(x_trn,y_trn,"Learners",t,"Coding","onevsone");
%Mdl = fitcecoc(x_trn,y_trn);
disp("End training")

%% Classification of trainign data

y_trn_est = predict(Mdl,x_trn);
correct = y_trn_est == y_trn;

mean(correct)

%% Classification of validation data

y_val_est = predict(Mdl,x_val);
correct = y_val_est == y_val;

mean(correct)

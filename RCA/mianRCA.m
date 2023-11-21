clear all;
close all;
clc;

load ('./cov_1');
cov1 = input;
load ('./cov_2');
cov2 = input;
load ('./cov_3');
cov3 = input;
load ('./cov_4');
cov4 = input;
load ('./cov_5');
cov5 = input;

channel_number = 22

[m,n] = size(cov1);
Covs1{m,1} = [];
for i = 1:m
    x = reshape(cov1(i,:),channel_number,channel_number);
    Covs1{i,1} = x;
end
Covs2{m,1} = [];
for i = 1:m
    x = reshape(cov2(i,:),channel_number,channel_number);
    Covs2{i,1} = x;
end
Covs3{m,1} = [];
for i = 1:m
    x = reshape(cov3(i,:),channel_number,channel_number);
    Covs3{i,1} = x;
end
Covs4{m,1} = [];
for i = 1:m
    x = reshape(cov4(i,:),channel_number,channel_number);
    Covs4{i,1} = x;
end
Covs5{m,1} = [];
for i = 1:m
    x = reshape(cov5(i,:),channel_number,channel_number);
    Covs5{i,1} = x;
end


Covs = [Covs1,Covs2,Covs3,Covs4,Covs5];


%data alignment
[CovsDA,D] =  RCA(Covs);
CovsDA1 = CovsDA(:,1);
CovsDA2 = CovsDA(:,2);
CovsDA3 = CovsDA(:,3);
CovsDA4 = CovsDA(:,4);
CovsDA5 = CovsDA(:,5);


X_1 = reshape(cell2mat(CovsDA1)',[size(CovsDA1{1}) numel(CovsDA1)]);
X_2 = reshape(cell2mat(CovsDA2)',[size(CovsDA2{1}) numel(CovsDA2)]);
X_3 = reshape(cell2mat(CovsDA3)',[size(CovsDA3{1}) numel(CovsDA3)]);
X_4 = reshape(cell2mat(CovsDA4)',[size(CovsDA4{1}) numel(CovsDA4)]);
X_5 = reshape(cell2mat(CovsDA5)',[size(CovsDA5{1}) numel(CovsDA5)]);


%Compute the Riemannian mean
M{1} = RiemannianMean(X_1);
M{2} = RiemannianMean(X_2);
M{3} = RiemannianMean(X_3);
M{4} = RiemannianMean(X_4);
M{5} = RiemannianMean(X_5);
D    = RiemannianMean(cat(3, M{1}, M{2}, M{3}, M{4}, M{5}));


%To vectors
newcov_1 = CovsToVecs(X_1,D);
newcov_2 = CovsToVecs(X_2,D);
newcov_3 = CovsToVecs(X_3,D);
newcov_4 = CovsToVecs(X_4,D);
newcov_5 = CovsToVecs(X_5,D);

input = real(newcov_1');
save('./vector_1.mat','input');
input = real(newcov_2');
save('./vector_2.mat','input');
input = real(newcov_3');
save('./vector_3.mat','input');
input = real(newcov_4');
save('./vector_4.mat','input');
input = real(newcov_5');
save('./vector_5.mat','input');

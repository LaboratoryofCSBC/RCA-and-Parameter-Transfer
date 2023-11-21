clear all;
close all;
clc;

load subject_1; %Load filtered data

channel_number = 22
feature_dimension = 750

data=input;
[m,n]=size(data);

newdata=[];
for i=1:m
    x=reshape(data(i,:),feature_dimension,channel_number);
    x=x';
    matrix=x*x'/(feature_dimension-1);
%     matrix=cov(x');
    xx=reshape(matrix',1,channel_number*channel_number);
    newdata=[newdata;xx];
    end
input=newdata;

a=input(2,:,:);
a=reshape(a,channel_number,channel_number);

save('./cov_1.mat','input');
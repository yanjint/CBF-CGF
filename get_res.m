clc
clear 
clc
close all
warning off
addpath(genpath('./datasets'))
addpath('./Funs');
addpath('./finchpp')
data = {'MSRC-v6'};  % MSRC  ORL  WikipediaArticles_total
orders = [1];    % 图滤波器
k = 5;           % 5阶稀疏构图 
lambda = [1]
for i= 1:length(lambda)
    for idx = 1: length(data) 
        load(data{idx})
    %%  运行数据据Yale_32x32，COIL20添加 
%             X = cell(1,2);
%             X{1} = feature;
%             X{1} = feature;
%             Y = label';
    %
    %针对Mfeat
%         X = data;
%         Y = truelabel{1};
%         for t=1:length(X)
%             X{t}=data{t}';
%         end
        for j=1:length(X)
            colnum=size(X{j},2);           % colnum = 24
            mole = repmat(std(X{j},0,2),1,colnum);
            mole(mole==0) = 1; 
            X{j}=(X{j}-repmat(mean(X{j},2),1,colnum))./mole;
        end
        
        c = length(unique(Y));   % 类别 classes = 7
        [G, K, F, evs]=prepare(X, k, c);
        KF = cross_con(K, F, 1);
        tic;
        [Y_g, y0, it, obj] = main_max(KF, G , c, c, 40, eps - evs,lambda(i));
        total_time = toc;
        fprintf('time = %d',total_time)
        result2(idx, :) = ClusteringMeasure_new(Y, Y_g)
    end
end



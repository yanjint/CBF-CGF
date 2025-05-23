function [Graph, K, F, sml_ev]=prepare(X, k, dim) %k表示阶数，dim表示类别
Graph = generate_graph(X, k);
num = size(Graph{1}, 1);    % 210
m=length(Graph);              % 5
% 视图 m = 5
% 得到graph的最小特征值
%
for i = 1 : m
    try
        evs(i) = eigs(Graph{i}, 1,'smallestabs');
    catch
        evss = eig(Graph{i});
        evs(i) = min(evss);
    end
end
sml_ev = min(evs);


F = cell(1, m);
L = cell(1, m);
for i=1:m
    graph = Graph{i};   %   以邻接矩阵的形式来体现graph
    L{i} = diag(sum(graph)) - graph; %得到拉普拉斯矩阵
    tmp = eig1(L{i}, dim+1, 0, 1);
    tmp(:, 1) = []; %第一列没意义
    tmp = tmp./repmat(sqrt(sum(tmp.^2,2)),1,dim);
    tmp(isnan(tmp))=0;
    F{i} = tmp;  %得到谱嵌入
%     F{i} = eig1(L{i}, dim, 0, 1);
end
K = cell(1, m);
for i = 1 :m
    K{i} = (eye(num) - L{i} / 2)^(1);
end
end
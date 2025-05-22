%% 灏嗗嵎绉牳涓庨噸鏋勫浘鍒嗗紑瀛︿範
function [Y_graph, y0, it, obj] = main_max(KF, Graph,c, dim_W, niter, evs,lambda)
num = size(KF{1}, 1);   % 210
m = size(KF, 1);     %5      
n = length(Graph);      %number of graphs
dim_F = size(KF{1}, 2);

% 文件写入
% fid=fopen('./record.txt','w+');
% fprintf(fid,'%s\n',string(5));
% fclose(fid);

%% init
Y = full(Init_Y(Graph, c)); %得到指示矩阵
[~, y0] = max(Y, [], 2);  %y0为簇号
%W的初始化
W = cell(m, m);
for i = 1 : m
    for j = 1 : m
        W{i,j} = eye(dim_F, dim_W) / m; %在分块矩阵中，m^2*c的情况下，单位正交 
    end 
end

a = ones(n + 1) / sqrt(n+1);   %论文中的q
a(end)=lambda;

% 初始化h
h = ones(n) / sqrt(n);

%初始化θ
T = ones(1,2*c);
T_norm = T ./ sqrt(2*c);
M = T_norm(1:c);
N = T_norm(c+1:2*c);
M = diag(M);    % θ1
N = diag(N);    % θ2
sita = cell(1,2);
sita{1} = M;
sita{2} = N;

% 初始化p
p = ones(m, m) ./ sqrt(m * m);

%初始化Y
YYYY = (Y * (Y' * Y*sita{1})^(-1)) * Y';
p_flatten = reshape(p, [1, m * m]); % 直接初始化的Vec(p)
Q = zeros(num);  % \hat(A)
for i = 1 : n
    Q = Q + Graph{i} * a(i);
end
Q = Q + YYYY * a(end);  %原图层面

G = zeros(num, dim_W);
for i = 1 : m
    for j = 1 : n
        G = G + p(i,j) * KF{i,j} * W{i,j}; %U_p*W，谱嵌入层面
    end
end

S2 = zeros(num);
for i = i :n
    S2 = S2 + Graph{i}*h(i);
end

% fid=fopen('./ORL.txt','w+');
for it=1:niter    % 40  
    %% update W
    % KF是列标准化的U
    fprintf('****这是第%d轮****\n',it)
    XX = reshape(KF, [1, m * m]);
    K = [];
    for i = 1 : m * m
        K = [K, XX{i}]; %没有p的U
    end
    H = [];
    for i = 1 : m * m
        H = blkdiag(H, p_flatten(i) * eye(dim_F)); %相当于一个权重W
    end
%     H = ((K * H)' * Q) * K * H;
    KH = K * H;
    H = (KH' * Q) * KH;
  

    F_H = eig1(H, dim_W, 1);
    W_flatten = cell(1, m*m);
    for i = 1 : m * m
        W_flatten{i} = F_H((i - 1) * dim_F + 1: i * dim_F, :);
    end
    W = reshape(W_flatten, [m, m]);


    %% update a
    beta = zeros(1, n + 1);

    for j = 1 : n
        beta(j) = trace((G' * Graph{j}) * G + evs * (G' * G));
    end
    beta(n + 1) = trace(G' * (YYYY + evs * eye(num)) * G);
    a = beta ./ norm(beta, 2);
    
%% update h
    beta_h = zeros(1,n);
    for j  = 1:n
        beta_h(j) = trace( Graph{j}*Y*(Y'*Y*sita{2})^(-1)*Y' +eps);
    end
    h = beta_h ./ norm(beta_h, 2);

    %% update  θ
    sita = cell(1,2);
    e_k1 = zeros(1,dim_F);
    for k=1:dim_F
        e_k1(k) = Y(:,k)'*G*G'*Y(:,k) / (Y(:,k)'*Y(:,k));
    end
    sita{1} = diag(norm(e_k1,2) ./ e_k1);  
    
    e_k2 = zeros(1,dim_F);
    for k=1:dim_F
        e_k2(k) = Y(:,k)'*S2*Y(:,k) / (Y(:,k)'*Y(:,k));
    end
    sita{2} = diag(norm(e_k2,2) ./ e_k2);  
    
  
    %% update p
    Q = zeros(num); %hat_A
    for i = 1 : n  
        Q = Q + Graph{i} * a(i);
    end
    Q = Q + YYYY * a(end);

    D = cell(m, m);
    for i = 1 : m
        for j = 1 : m
            D{i,j} = KF{i, j} * W{i, j}; %UW
        end
    end
    D_flatten = reshape(D, [1, m * m]);
%     U = cell(1, m * m);
%     for i = 1 : m * m
%         U{i} = Q * D_flatten{i};
%     end
    U = cellfun(@(x) Q * x, D_flatten, 'UniformOutput', 0);
    A = Vec(D_flatten);
    B = Vec(U);
    p_flatten = eig1(A'*B, 1, 1, 0);
    p = reshape(p_flatten, [m, m]);

    %% update Y
   

    G = zeros(num, dim_W);
    for i = 1 : m
        for j = 1 : n
            G = G + p(i,j) * KF{i,j} * W{i,j}; %U_p*W，谱嵌入层面
        end
    end
    
    %得到原图层面的嵌入
    S2 = zeros(num); %hat_A
    for i = 1 : n  
        S2 = S2 + Graph{i} * h(i);
    end
    
    KK =zeros(num);
    for i=1:c
        KK = (sita{1}(i,i))^(-1)*G*G' + (sita{2}(i,i))^(-1)*S2;
    end
    KK = (KK+KK')/2;

    %Y = CDKM(G*G', Y,S2,sita);
    Y = solve_F(KK, Y);
    %Y = full(ind2vec(y'))';
    [~, Y_1] = max(Y, [], 2);
%     Y = coordinate_descend(G * G', Y);
    YYYY = (Y * (Y' * Y*sita{1})^(-1)) * Y'+(Y * (Y' * Y*sita{2})^(-1)) * Y' ;
     obj(it) = get_obj(Y, a, p, W, KF, Graph,h,sita,lambda);
     %fprintf(fid,'%s\n',string(obj(it)));
   if it>2 && (obj(it) -obj(it-1)) / obj(it-1) < 1e-4
       break
   end
end

[~, Y_graph] = max(Y, [], 2);
% fclose(fid);
end


function B=Vec(A)
m=length(A);
[d1,d2]=size(A{1});
B=zeros(d1*d2,m);
for j=1:m
    B(:,j)=A{j}(:);
end
end


function obj = get_obj(Y, a, p, W, KF, Graph,h,sita,lambda)
m = size(KF, 1); %5
num = size(KF{1}, 1); %210
G = zeros(num, size(W{1}, 2));
for i = 1 : m
    for j = 1 : m
        G = G + p(i,j) * KF{i,j} * W{i,j};
    end
end
%原图层面
Q = zeros(num);
for i = 1 : length(Graph)
    Q = Q + Graph{i} * a(i);
end

obj1 = trace((G' * Q) * G);

S2 = zeros(num);
for i=i:length(Graph)
    S2 = S2 + Graph{i}*h(i);
end

YYYY1 = (Y * (Y' * Y*sita{1})^(-1)) * Y';
YYYY2 = (Y * (Y' * Y*sita{2})^(-1)) * Y';
obj2 = trace(G*G'*YYYY1 + S2*YYYY2) ;

obj = obj1 + lambda * obj2;
end
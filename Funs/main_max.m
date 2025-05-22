function [Y_graph, y0, it, obj] = main_max(KF, Graph,c, dim_W, niter, evs,lambda)
num = size(KF{1}, 1);   
m = size(KF, 1);     
n = length(Graph);     
dim_F = size(KF{1}, 2);



%% init
Y = full(Init_Y(Graph, c)); %Indicator matrices
[~, y0] = max(Y, [], 2);  %
%
W = cell(m, m);
for i = 1 : m
    for j = 1 : m
        W{i,j} = eye(dim_F, dim_W) / m; 
    end 
end

a = ones(n + 1) / sqrt(n+1);   
a(end)=lambda;

%h
h = ones(n) / sqrt(n);

%θ
T = ones(1,2*c);
T_norm = T ./ sqrt(2*c);
M = T_norm(1:c);
N = T_norm(c+1:2*c);
M = diag(M);    % θ1
N = diag(N);    % θ2
sita = cell(1,2);
sita{1} = M;
sita{2} = N;

% p
p = ones(m, m) ./ sqrt(m * m);

%Y
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


for it=1:niter    % 40  
    %% update W
    % KF是列标准化的U
    fprintf('****这是第%d轮****\n',it)
    XX = reshape(KF, [1, m * m]);
    K = [];
    for i = 1 : m * m
        K = [K, XX{i}]; %
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
    Q = zeros(num); 
    for i = 1 : n  
        Q = Q + Graph{i} * a(i);
    end
    Q = Q + YYYY * a(end);

    D = cell(m, m);
    for i = 1 : m
        for j = 1 : m
            D{i,j} = KF{i, j} * W{i, j}; %
        end
    end
    D_flatten = reshape(D, [1, m * m]);

    U = cellfun(@(x) Q * x, D_flatten, 'UniformOutput', 0);
    A = Vec(D_flatten);
    B = Vec(U);
    p_flatten = eig1(A'*B, 1, 1, 0);
    p = reshape(p_flatten, [m, m]);

    %% update Y
   

    G = zeros(num, dim_W);
    for i = 1 : m
        for j = 1 : n
            G = G + p(i,j) * KF{i,j} * W{i,j}; 
        end
    end
    

    S2 = zeros(num); 
    for i = 1 : n  
        S2 = S2 + Graph{i} * h(i);
    end
    
    KK =zeros(num);
    for i=1:c
        KK = (sita{1}(i,i))^(-1)*G*G' + (sita{2}(i,i))^(-1)*S2;
    end
    KK = (KK+KK')/2;
    Y = solve_F(KK, Y);
    YYYY = (Y * (Y' * Y*sita{1})^(-1)) * Y'+(Y * (Y' * Y*sita{2})^(-1)) * Y' ;
     obj(it) = get_obj(Y, a, p, W, KF, Graph,h,sita,lambda);
   if it>2 && (obj(it) -obj(it-1)) / obj(it-1) < 1e-4
       break
   end
end

[~, Y_graph] = max(Y, [], 2);
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
m = size(KF, 1); 
num = size(KF{1}, 1); 
G = zeros(num, size(W{1}, 2));
for i = 1 : m
    for j = 1 : m
        G = G + p(i,j) * KF{i,j} * W{i,j};
    end
end

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
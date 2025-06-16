function [Y, minO, iter_num, obj] = CDKM1(X, F,S2,sita)
% Input
% X d*n data
% label is initial label n*1
% c is the number of clusters
% Output
% Y is the label vector n*1
% minO is the Converged objective function value
% iter_num is the number of iteration
% obj is the objective function value

[~, label] = max(F, [], 2); % label表示列号
[n,c] = size(F); % 210*7
last = 0;
iter_num = 0;
%% compute Initial objective function value
for ii=1:c
        idxi = find(label==ii); %类别i的下标
        Xi = X(:, idxi);     %得到类别i的特征矩阵
        ceni = mean(Xi,2);  %对类别i的特征进行求均值
        center(:,ii) = ceni; %将均值放入质心矩阵
        %通过欧式距离来计算
        c2 = ceni'*ceni; 
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
        sumd(ii,1) = sum(d2c); 
end
obj(1)= sum(sumd);    % Initial objective function value
%% store once
for i=1:n
    XX(i)=X(:,i)'* X(:,i); %相当于S1的迹
end    
BB = X*F;
aa=sum(F,1); %表示每个种类的个数，相当于y'*y
FXXF=BB'*BB;% F'*X'*X*F;

%得到S2的相关信息
S2_tr = F'*S2*F ;%%
S2_XX = diag(S2);


while any(label ~= last)   
    last = label;   

 for i = 1:n   
     m = label(i) ; %m表示类别
    if aa(m)==1
        continue;  
    end 
    for k = 1:c        
        if k == m   
           V1(k) = FXXF(k,k)- 2 * X(:,i)'* BB(:,k) + XX(i);
           V3(k) = S2_tr(k,k) - 2 * F(:,k)'*S2(:,i) + S2_XX(i);
           delta(k) = (sita{1}(k,k))^(-1)*(FXXF(k,k) / aa(k) - V1(k) / (aa(k) -1))+(sita{2}(k,k))^(-1)*(S2_tr(k,k) / aa(k) -V3(k) /(aa(k)-1)); 
        else  
           V2(k) =(FXXF(k,k)  + 2 * X(:,i)'* BB(:,k) + XX(i));
           V4(k) =  S2_tr(k,k) + 2 * F(:,k)'*S2(:,i) + S2_XX(i);
           delta(k) = (sita{1}(k,k))^(-1)*(V2(k) / (aa(k) +1) -  FXXF(k,k)  / aa(k)) + (sita{2}(k,k))^(-1)*(V4(k) /(aa(k)+1) -S2_tr(k,k) / aa(k)); 
        end         
    end  
    [~,q] = max(delta);     
    if m~=q        
         BB(:,q)=BB(:,q)+X(:,i); % BB(:,p)=X*F(:,p);
         BB(:,m)=BB(:,m)-X(:,i); % BB(:,m)=X*F(:,m);
         aa(q)= aa(q) +1; %  FF(p,p)=F(:,p)'*F(:,p);
         aa(m)= aa(m) -1; %  FF(m,m)=F(:,m)'*F(:,m)      
         FXXF(m,m)=V1(m); 
         FXXF(q,q)=V2(q);

         
         
         S2_tr(m,m) = V3(m);
         S2_tr(q,q) = V4(q);
         
         
         label(i)=q;
    end
 end 
 
 
  iter_num = iter_num+1;
%% compute objective function value
   for ii=1:c
        idxi = find(label==ii);
        Xi = X(:,idxi);     
        ceni = mean(Xi,2);   
        center1(:,ii) = ceni;
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi; 
        sumd(ii,1) = sum(d2c); 
    end
    obj(iter_num+1) = sum(sumd) ;     %  objective function value     
end    
 minO=min(obj);
 Y=label;
end

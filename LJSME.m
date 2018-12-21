function [U obj]=LJSME(X,Wb,Ww,k,gnd,alpha,beta)
%--------------------------------------------------------------------------
% title: Locally Joint Sparse Marginal Embedding for Feature Extraction
% min Tr(U'*Sw*U-alpha*U'*Sb*U)+beta*||U||21          
% s.t. U'*U=I
%--------------------------------------------------------------------------
%% Input
% X:data matrix, each row is a data point
% Wb: weight matrix corresponding to between class
% Ww: weight matrix corresponding to within class
% k:the number of the projections
% gnd: label vector
% alpha,beta: parameters to balance the objective function

%% Output
% U:projection matrix
% obj:the value of the objective function

% Dongmei Mo, Zhihui Lai*, Waikeung Wong
% 21 Dec. 2018
% The Hong Kong Polytechnic University
%--------------------------------------------------------------------------

ite_max= 30;
clear A0;

if (~exist('k','var'))
   k=size(X,2);
end

if k>size(X,2)
    k=size(X,2);
end

[A d v]=svd(X','econ');
A0=A(:,1:k);  
n = size(A0,1);
du = zeros(n,1);
for i=1:n
    du(i)=0.5./norm(A0(i,:));
end
Du = diag(du);
for ite=1:ite_max
    Sw=zeros(size(X,2)); Sb=zeros(size(X,2));       
    msmp = size(X,1);
    for i= 1:msmp
        for j= 1:msmp
            if Ww(i,j)~=0
                Xw_diff = (X(i,:)-X(j,:)).*Ww(i,j);
                Xw_U = Xw_diff*A0;
                if sqrt(sum(Xw_U.*Xw_U,2))==0
                    dw_i= 10^8;
                else
                    dw_i = 0.5./sqrt(sum(Xw_U.*Xw_U,2));
                end
                Sw = Sw + Xw_diff'.*dw_i*Xw_diff;
            end

             if Wb(i,j)~=0
                Xb_diff = (X(i,:)-X(j,:)).*Wb(i,j);
                Xb_U = Xb_diff*A0;
                if sqrt(sum(Xb_U.*Xb_U,2))==0
                    db_i=  10^8;
                else
                    db_i = 0.5./sqrt(sum(Xb_U.*Xb_U,2));
                end
                Sb = Sb + Xb_diff'*db_i*Xb_diff;
            end

        end
    end
    obj(ite)=trace( A0'*(Sw - alpha*Sb + beta*Du )*A0);
    [vec val0]=eig(Sw - alpha*Sb + beta*Du);
    [val ind]=sort(diag(val0),'ascend');
    A0=real(vec(:,ind(1:k)));
    for i=1:size(A0,1)
        du(i)=0.5./norm(A0(i,:));
    end
    Du = diag(du);
    U=A0;
end
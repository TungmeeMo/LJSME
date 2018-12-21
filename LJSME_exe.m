clear; clc ;
load('.\dataset\Xpca.mat')
load('.\dataset\Xtpca.mat')
load('.\dataset\Y.mat')
load('.\dataset\Xtrain_label.mat')
load('.\dataset\Xtest_label.mat')
K = 5;
lambda1 = -3;
lambda2 = -3;
k = 150;
rate_AR_raw= KNN_Classfier(Xpca, Xtrain_label, Xtpca,Xtest_label, 1);

[Wb, Ww]= C_LDA_local_Wb_Ww(Xpca',Xtrain_label,K,0);
[B,objValue] = LJSME(Xpca',Wb,Ww,k,lambda2,10^lambda1,10^lambda2);
 plot(B);
 for dim=size(B,2)
        Ytrain=B(:,1:dim)'*Xpca;
        Ytest=B(:,1:dim)'*Xtpca;%
        rate_AR_LJSME= KNN_Classfier(Ytrain, Xtrain_label, Ytest,Xtest_label, 1);
 end
fprintf('\n==================================================================\n');
fprintf('The face recognition rate of raw data on AR database is %s\n', num2str(rate_AR_raw,'%.2f'));
fprintf('The face recognition rate of LJSME on AR database is %s\n', num2str(rate_AR_LJSME,'%.2f'));
fprintf('==================================================================\n');

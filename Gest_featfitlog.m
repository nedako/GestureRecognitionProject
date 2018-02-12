function [criterion] = Gest_featfitlog(X_train,y_train,X_test,y_test)


sp = categorical(y_train);
[B,~,~] = mnrfit(X_train,sp);
pihat = mnrval(B,X_test);
[~ , Gpred] = max(pihat, [] , 2);
criterion = sum(y_test ~= Gpred);

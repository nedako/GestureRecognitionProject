function [criterion] = Gest_featfitNaiveB(X_train,y_train,X_test,y_test)

Mdl  = fitcnb(X_train,y_train,'Distribution',repmat({'kernel'} , 1,size(X_train , 2)) , 'kernel' , 'normal');

[Gpred,Posterior,~] = predict(Mdl,X_test);
criterion = sum(y_test ~= Gpred);
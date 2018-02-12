function [Posterior , criterion , Mdl] = Gest_fitLDA(X_train,y_train,X_test,y_test)

 Mdl = fitcdiscr(X_train,y_train);
[Gpred,Posterior,~] = predict(Mdl,X_test);
criterion = sum(y_test ~= Gpred);
function [Posterior , criterion , Mdl] = Gest_fitNaiveB(X_train,y_train,X_test,y_test ,prior)
if isempty(prior)
    Mdl  = fitcnb(X_train,y_train,'Distribution',repmat({'kernel'} , 1,size(X_train , 2)) , 'kernel' , 'normal');
else
    Mdl  = fitcnb(X_train,y_train,'Distribution',repmat({'kernel'} , 1,size(X_train , 2)) , 'kernel' , 'normal' , 'Prior' , prior);
end
[Gpred,Posterior,~] = predict(Mdl,X_test);
criterion = sum(y_test ~= Gpred);
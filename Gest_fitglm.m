function out = Gest_fitglm(X , Y , CV)



folds = unique(CV);
Mdl.B = zeros(min(size(X)) , min(size(Y)));
for f = 1:length(folds)
    id = CV == folds(f);
    X_train = X(~id , :);
    Y_train = Y(~id , :);
    X_test = X(id , :);
    Y_test = Y(id , :);
    
    %
    %     Mdl.B_f{f} =  pinv(X_train) * Y_train;
    %     Mdl.B = Mdl.B + Mdl.B_f{f};
    %     err = Y_test - (X_test * Mdl.B_f{f});
    %     Mdl.err(f) = mean(mean(err.^2));
    
    for gn = 1:min(size(Y_train))
        Mdl = fitglm(X_train(:,2:end),Y_train(:,gn));
        Gpred = predict(Mdl,X_test(:,2:end));
        err(f,gn) = mean(mean((Y_test(:,gn) - Gpred).^2));
        B(:,gn) = table2array(Mdl.Coefficients(2:end , 1));
    end
    
end
Mdl.B  = Mdl.B/f;
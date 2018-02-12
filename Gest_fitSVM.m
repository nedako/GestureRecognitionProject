function [PosteriorRegion , criterion, Mdl] = Gest_fitSVM(X_train,y_train,X_test,y_test)

t = templateSVM('Standardize',1,'KernelFunction','gaussian');
Mdl = fitcecoc(X_train,y_train,'Learners',t,'ClassNames',[1 2 3 4 5 6],'FitPosterior',1);
[~,~,~,PosteriorRegion] = predict(Mdl,X_test);
[~ , Gpred] = max(PosteriorRegion, [] , 2);
criterion = sum(y_test ~= Gpred);
function out = Gest_featureSelect(D , what , baseDir, FeatureFieldName)

% heads-up --> logistic takes a loooooong time

eval(['F = D.' , FeatureFieldName , ';']);


switch what
    case 'logistic'
        % ===============================  sequential feature selection using multinomial logistic regression
        c = cvpartition(V.GestNum,'k',10);
        fun = @Gest_featfitlog;
        opts = statset('display','iter');
        [fs,history] = sequentialfs(fun,F,V.GestNum,'cv',c,'options',opts);
    case 'NaiveB'
        % ===============================  sequential feature selection using multiclass Naive Bayes Classifier
        c = cvpartition(D.GestNum,'k',5);
        fun = @Gest_featfitNaiveB;
        opts = statset('display','iter');
        [out.fs,out.history] = sequentialfs(fun,F,D.GestNum,'cv',c,'options',opts);
        save([baseDir , '/Crossvalidated_NB_Selected_Features_Distance.mat'] ,'fs' , 'history');
     
end
out.fs = fs;
out.history = history;

                

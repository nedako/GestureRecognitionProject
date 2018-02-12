function [ConfMat] = Gest_test(baseDir , what , viz)
%% [ConfMat] = Gest_test(baseDir , what);
% Inputs:
%   baseDir : where the data are saved
%   what : classifier

%          'logistic'
%          'logisticPartial'
%          'NaiveBayes'
%          'LDA'
%   viz : 0 or 1 (visualization)
%
%
% Output:
%   Confusion Matrix of the classification
%   Also plots the confusion matrix and the ROC
%
% Neda Kordjazi
% November 2017
%%
load([baseDir , '/Gest/Crossvalidated_NB_Selected_Features_Distance.mat']);
load([baseDir , '/Gest/Crossvalidated_loss_LDA_interp_Distance.mat']);
[D ,isBad] = Gest_getdata(baseDir , 'All');
for i = 1:length(D.GestNum)
    D.interpEMG{i,1} = interp1([1:50] ,D.EMG{i}, linspace(1,50 , bestInterp),'spline');
    D.interpDist(i , :) = pdist(transpose(D.interpEMG{i}) , 'euclidean');
end

X_test = D.interpDist;
y_test = D.GestNum;
switch what
    case 'logistic'
        load([baseDir , '/Gest/EndClassifiers/END_LogisticClassifier_Distance.mat']);
        Prob = mnrval(B,X_test);
        
        [~ , Gest]   = max(Prob , [],2);
        ConfMat = confusionmat(y_test,Gest);
        ConfMat =  100*bsxfun(@rdivide, ConfMat , sum( ConfMat , 2));
        
        Acc = sum(Gest==y_test)/length(y_test);
        disp(['Classifier completed!'])
        disp(['Average Accuracy is ' , num2str(Acc)]);
        
        % ROC plot
        if viz
            figure('color' , 'white')
            subplot(121)
            hold on
            a = unique(y_test);
            for g = 1:length(a)
                [X,Y] = perfcurve(y_test,max(Prob , [],2),a(g));
                plot(X , Y, 'LineWidth' , 3 );
                grid on
                set(gca , 'FontSize' , 16, 'XTick' , [0 .5 1], 'YTick' , [0 .5 1]);
                title(['Test ROC plot Logistic Regression'])
                ylabel('True Positive rate')
                xlabel('False Positive rate')
                hold on
                axis square
            end
            legend({'G1' , 'G2' , 'G3' , 'G4' , 'G5' , 'G6'})
            line([0 1], [0 1], 'LineWidth' , 2,'LineStyle' , ':' , 'color' , 'r')
            
            
            % plot confusion matrix
            M = ConfMat;
            subplot(122)
            imagesc(M)
            axis square
            xlabel('Gesture')
            ylabel('Gesture')
            set(gca , 'FontSize' , 16, 'XTick' , [1:6], 'YTick' , [1:6]);
            title('Test Confusion Matrix - Logistic Classification on all the Distance')
            colorbar
        end
    case 'logisticPartial'
        load([baseDir , '/Gest/EndClassifiers/END_LogisticClassifier_Distance.mat']);
        Prob = mnrval(B,X_test);
        
        [~ , Gest]   = max(Prob , [],2);
        ConfMat = confusionmat(y_test,Gest);
        ConfMat =  100*bsxfun(@rdivide, ConfMat , sum( ConfMat , 2));
        
        Acc = sum(Gest==y_test)/length(y_test);
        disp(['Classifier completed!'])
        disp(['Average Accuracy is ' , num2str(Acc)]);
        
        if viz
            % ROC plot
            figure('color' , 'white')
            subplot(121)
            hold on
            a = unique(y_test);
            for g = 1:length(a)
                [X,Y] = perfcurve(y_test,max(Prob , [],2),a(g));
                plot(X , Y, 'LineWidth' , 3 );
                grid on
                set(gca , 'FontSize' , 16, 'XTick' , [0 .5 1], 'YTick' , [0 .5 1]);
                title(['Test ROC plot Logistic Regression'])
                ylabel('True Positive rate')
                xlabel('False Positive rate')
                hold on
                axis square
            end
            legend({'G1' , 'G2' , 'G3' , 'G4' , 'G5' , 'G6'})
            line([0 1], [0 1], 'LineWidth' , 2,'LineStyle' , ':' , 'color' , 'r')
            
            
            % plot confusion matrix
            M = ConfMat;
            subplot(122)
            imagesc(M)
            axis square
            xlabel('Gesture')
            ylabel('Gesture')
            set(gca , 'FontSize' , 16, 'XTick' , [1:6], 'YTick' , [1:6]);
            title('Test Confusion Matrix - Logistic Classification on all the Distance')
            colorbar
        end
    case 'NaiveBayes'
        load([baseDir , '/Gest/EndClassifiers/END_NaiveBayesClassifier_Distance.mat']);
        [Gpred,Prob,~] = predict(Mdl,X_test);
        
        
        [~ , Gest]   = max(Prob , [],2);
        ConfMat = confusionmat(y_test,Gest);
        ConfMat =  100*bsxfun(@rdivide, ConfMat , sum( ConfMat , 2));
        
        Acc = sum(Gest==y_test)/length(y_test);
        disp(['Classifier completed!'])
        disp(['Average Accuracy is ' , num2str(Acc)]);
        if viz
            % ROC plot
            figure('color' , 'white')
            subplot(121)
            hold on
            a = unique(y_test);
            for g = 1:length(a)
                y_test  = D.GestNum;
                [X,Y] = perfcurve(y_test,max(Prob , [],2),g(a));
                plot(X , Y, 'LineWidth' , 3 );
                grid on
                set(gca , 'FontSize' , 16, 'XTick' , [0 .5 1], 'YTick' , [0 .5 1]);
                title(['Test ROC plot Naive Bayes'])
                ylabel('True Positive rate')
                xlabel('False Positive rate')
                hold on
                axis square
            end
            legend({'G1' , 'G2' , 'G3' , 'G4' , 'G5' , 'G6'})
            line([0 1], [0 1], 'LineWidth' , 2,'LineStyle' , ':' , 'color' , 'r')
            
            
            % plot confusion matrix
            M = ConfMat;
            subplot(122)
            imagesc(M)
            axis square
            xlabel('Gesture')
            ylabel('Gesture')
            set(gca , 'FontSize' , 16, 'XTick' , [1:6], 'YTick' , [1:6]);
            title('Test Confusion Matrix - Naive Bayes Classification')
            colorbar
        end
    case 'LDA'
        
        load([baseDir , '/Gest/EndClassifiers/END_LDAClassifier_Distance.mat']);
        [Gpred,Prob,~] = predict(Mdl,X_test);
        
        
        [~ , Gest]   = max(Prob , [],2);
        ConfMat = confusionmat(y_test,Gest);
        ConfMat =  100*bsxfun(@rdivide, ConfMat , sum( ConfMat , 2));
        
        Acc = sum(Gest==y_test)/length(y_test);
        disp(['Classifier completed!'])
        disp(['Average Accuracy is ' , num2str(Acc)]);
        
        % ROC plot
        if viz
            figure('color' , 'white')
            subplot(121)
            hold on
            a = unique(y_test);
            for g = 1:length(a)
                y_test  = D.GestNum;
                [X,Y] = perfcurve(y_test,max(Prob , [],2),a(g));
                plot(X , Y, 'LineWidth' , 3 );
                grid on
                set(gca , 'FontSize' , 16, 'XTick' , [0 .5 1], 'YTick' , [0 .5 1]);
                title(['Test ROC plot LDA'])
                ylabel('True Positive rate')
                xlabel('False Positive rate')
                hold on
                axis square
            end
            legend({'G1' , 'G2' , 'G3' , 'G4' , 'G5' , 'G6'})
            line([0 1], [0 1], 'LineWidth' , 2,'LineStyle' , ':' , 'color' , 'r')
            
            
            % plot confusion matrix
            M = ConfMat;
            subplot(122)
            imagesc(M)
            axis square
            xlabel('Gesture')
            ylabel('Gesture')
            set(gca , 'FontSize' , 16, 'XTick' , [1:6], 'YTick' , [1:6]);
            title('Test Confusion Matrix - LDA Classification')
            colorbar
        end
end
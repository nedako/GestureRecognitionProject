function out = Gest_analyze(what , baseDir , varargin)
%% out = Gest_analyze(what , baseDir , varargin)

% Inputs:
%   baseDir : where the data are saved
%   what : 
%           'Unpack'
%           'VisualizeSummary'
%       UNIVARIATE CASES
%           'Interp_class_per_window'
%               calculate and visualize the moving window intepolated
%               classification accuracy
%           'iterp_level_select' :
%               finds the best interpolation window for each mucsle
%           'CrossVal_MuscleEnsemble'
%               finds the cross validated accuracy o a logistic/LDA/NB
%               ensemble for each muscle
%       MUNIVARIATE CASES
%          'Interp_class_per_window_dist'
%               calculate and visualize the moving window intepolated
%               classification accuracy
%          'FeatSelect'
%               perform naive bayes sequenctial feature selectoin
%          'CrossVal_classify_dist'
%               finds the cross validated accuracy o a logistic/LDA/NB
%               classifiers for full/partial distance
%          'BuildAndTest_EndClassifier_dist'
%               build the final classifier on the crossvalidation set and
%               tst it on the test set
%       varargin:
%               attributes of different case...
%
%
% Output:
%   Confusion Matrix of the classification
%   Also plots the confusion matrix and the ROC
%
% Neda Kordjazi
% November 2017
%%

% Deal with varargin
c = 1;
while(c<=length(varargin))
    switch(varargin{c})
        case {'saveoutput'}
            eval([varargin{c} '= (varargin{c+1} >= 1);']);
            c=c+2;
        case {'saveName'}
            eval([varargin{c} '= (varargin{c+1});']);
            c=c+2;
        case {'interpSamp'}
            eval([varargin{c} '= varargin{c+1};']);
            c=c+2;
        case {'whichInterp'}
            eval([varargin{c} '= varargin{c+1};']);
            c=c+2;
        case {'Method'}
            eval([varargin{c} '= varargin{c+1};']);
            c=c+2;
        case {'AllFeatures'}
            eval([varargin{c} '= varargin{c+1};']);
            c=c+2;
        otherwise
            error(sprintf('No such option: %s',varargin{c}));
    end
    
end
if ~exist('whichInterp')
    whichInterp = 7;      
end

switch what
    case 'Unpack'
        % unpack the EMG data and amend the data structure with summaries of the data and save to disc
        load([baseDir , '/Gest/Data_All.mat']);

        % time intepolation
        for ip = 1:length(interpSamp)
            for i = 1:length(Data.GestNum)
                eval(['Data.interpEMG',num2str(interpSamp(ip)) , '{i,1} = interp1([1:50] ,Data.EMG{i}, linspace(1,50 , interpSamp(ip)),''spline'');']);
                eval(['Data.interpDist',num2str(interpSamp(ip)) , '(i , :) = pdist(transpose(Data.interpEMG',num2str(interpSamp(ip)) , '{i}) , ''euclidean'');']);
            end
        end
        
        
        % create balanced gesture CV folds
        % fold 11 is never used in the cross-validation reps - save for the end testing
        CVI = zeros(length(Data.TiralNum) , 1);
        for gn = 1:6
            id = Data.GestNum == gn;
            L = sum(id);
            CVI(id) = crossvalind('Kfold', L, 10);
        end
        save([baseDir , '/Gest/CrossValidationIdx.mat'] , 'CVI')
        save([baseDir , '/Gest/Data_all_amended.mat'] , 'Data')
        
        out = Data;
    case 'VisualizeSummary'
        load([baseDir , '/Gest/Data_all_amended.mat']);
        colorz = {[0 0  1],[1 0 0],[0 1 0],[1 0 1],[0 1 1],[0.7 0.7 0.7],[1 1 0],[.3 .3 .3]};
        Data1 = Data;
        clear Temp
        
        Temp.EMG = [];
        Temp.TiralNum = [];
        Temp.GestNum = [];
        Temp.timeStapm = [];
        Temp.interpTime = [];
        Temp.iGestNum = [];
        Temp.iTiralNum = [];
        eval('Temp.interpEMG = [];');
        for gnum = 1:6
            Data  = getrow(Data1 , Data1.GestNum == gnum);
            T.EMG      = cell2mat(Data.EMG);
            T.TiralNum = repmat(Data.TiralNum , 50 , 1);
            T.GestNum  = repmat(Data.GestNum , 50 , 1);
            T.timeStapm = repmat([1:50]' , length(Data.EMG) , 1);
            %========= create plotting indecies for interpolated data of whichInterp
            eval(['T.interpEMG      = cell2mat(Data.interpEMG',num2str(whichInterp),');']);
            T.interpTime  = repmat(linspace(1,50 , whichInterp)' , length(Data.EMG) , 1);
            T.iGestNum  = repmat(Data.GestNum , whichInterp , 1);
            T.iTiralNum = repmat(Data.TiralNum , whichInterp , 1);
            
            Temp = addstruct(Temp , T);
            clear T
        end
        
        
        
        All = tapply(Temp , {'iTiralNum' , 'iGestNum'} , {'interpEMG' , 'nanmedian(x)' , 'name' , 'medianiEMG'},...
            {'interpEMG' , 'nanmean(x)' , 'name' , 'meaniEMG'});
        
        figure('color' , 'white')
        for getnum = 1:6
            subplot(2,3,getnum)
            [coor_t , plt_t , err_t] = lineplot([Temp.timeStapm ], Temp.EMG , 'subset' , Temp.GestNum == getnum ,'style_shade',...
                'shadecolor' ,colorz , 'linecolor',colorz);
            xlabel('t')
            ylabel('\mu volts')
            title(['Gesture ' , num2str(getnum)])
        end
        
        
        figure('color' , 'white')
        for getnum = 1:6
            subplot(2,3,getnum)
            [coor_t , plt_t , err_t] = lineplot([Temp.interpTime ], Temp.interpEMG , 'subset' , Temp.iGestNum == getnum ,'style_shade',...
                'shadecolor' ,colorz , 'linecolor',colorz);
            xlabel('t')
            ylabel('\mu volts')
            title(['Interpolated - Gesture ' , num2str(getnum)])
        end
        
        
        figure('color' , 'white')
        for musnum = 1:8
            subplot(2,4,musnum)
            for getnum = 1:6
                [coor_t , plt_t , err_t] = lineplot([Temp.timeStapm], Temp.EMG(:,musnum) , 'plotfcn' , 'nanmedian' , 'subset' , Temp.GestNum == getnum , 'style_shade',...
                    'shadecolor' ,colorz{getnum} , 'linecolor',colorz{getnum});
                hold on
            end
            title(['Muscle ' , num2str(musnum)])
            grid on
            xlabel('t')
            ylabel('\mu volts')
        end
        
        figure('color' , 'white')
        for musnum = 1:8
            subplot(2,4,musnum)
            for getnum = 1:6
                [coor_t , plt_t , err_t] = lineplot([Temp.interpTime], Temp.interpEMG(:,musnum) , 'plotfcn' , 'nanmedian' , 'subset' , Temp.iGestNum == getnum , 'style_shade',...
                    'shadecolor' ,colorz{getnum} , 'linecolor',colorz{getnum} );
                hold on
            end
            
            title(['Interpolated - Muscle ' , num2str(musnum)])
            grid on
            xlabel('t')
            ylabel('\mu volts')
        end
        
        
        
        
        dumFig = figure;
        [coor_med , plt_med , err_med] = lineplot(All.iGestNum,All.medianiEMG);
        hold on
        [coor_m , plt_m , err_m] = lineplot(All.iGestNum,All.meaniEMG);
        close(dumFig)
        
        figure('color' , 'white')
        for m = 1:8
            subplot(121)
            errorbar(coor_m' , plt_m(m,:)' , err_m(m,:)' , 'LineWidth' , 3 , 'color' , colorz{m})
            hold on
            grid on
            title('Interpolated mean muscle activity variance in different gestures')
            set(gca , 'FontSize' , 16 , 'XLim' , [0 7] , 'XTick' , [1:6] , 'XTicklabel' , {'G1' , 'G2' , 'G3' , 'G4' , 'G5' , 'G6'});
            xlabel('Gesture')
            ylabel('\mu volts')
            
            subplot(122)
            errorbar(coor_med' , plt_med(m,:)' , err_med(m,:)' , 'LineWidth' , 3 , 'color' , colorz{m})
            hold on
            grid on
            title('Interpolated median muscle activity variance in different gestures')
            legend({'Muscle 1' , 'Muscle 2' , 'Muscle 3' , 'Muscle 4' , 'Muscle 5' , 'Muscle 6' 'Muscle 7' 'Muscle 8'})
            legend()
            set(gca , 'FontSize' , 16 , 'XLim' , [0 7] , 'XTick' , [1:6] , 'XTicklabel' , {'G1' , 'G2' , 'G3' , 'G4' , 'G5' , 'G6'});
            xlabel('Gesture')
            ylabel('\mu volts')
        end
        out = All;
    case 'Interp_class_per_window'
        load([baseDir , '/Gest/Data_All.mat']);
        load([baseDir , '/Gest/CrossValidationIdx.mat'])
        % exclude the test fold
        D = getrow(Data , CVI ~= 11);
        
        for ip = 1:length(interpSamp)
            for i = 1:length(D.GestNum)
                D.interpEMG{i,1} = interp1([1:50] ,D.EMG{i}, linspace(1,50 , interpSamp(ip)),'spline');
            end
            
            Temp.interpEMG = [];
            Temp.interpTime = [];
            Temp.iGestNum = [];
            eval('Temp.interpEMG = [];');
            for gnum = 1:6
                D1 = getrow(D , D.GestNum == gnum);
                T.interpEMG      = cell2mat(D.interpEMG);
                T.interpTime  = repmat(linspace(1,50 , interpSamp(ip))' , length(D.EMG) , 1);
                T.iGestNum  = repmat(D.GestNum , interpSamp(ip) , 1);
                Temp = addstruct(Temp , T);
                clear T
            end
            
            
            time{ip} = unique(Temp.interpTime);
            for t = 1:length(time{ip})
                C = getrow(Temp , Temp.interpTime == time{ip}(t));
                c = cvpartition(C.iGestNum,'k',10);
                Mdl = fitcdiscr(C.interpEMG , C.iGestNum,'CV' , c);
                loss(ip , t) = kfoldLoss(Mdl);
                disp(['Cross-Validated LDA - interpolation constant is  ' , num2str(interpSamp(ip)) ,' completed!'])
            end
        end
        L = zeros(size(loss));
        for ip = 1:length(interpSamp)
            ind = round(linspace(1,50 , interpSamp(ip)));
            for j = 1:length(ind)-1
                L(ip , ind(j) : ind(j+1)-1) = repmat(loss(ip,j) , 1  , ind(j+1) - ind(j));
            end
        end
        save([baseDir , '/Gest/CrossValidated_MovingWindow_Loss.mat'] , 'L' , 'interpSamp')
        figure('color', 'white')
        imagesc(flipud(L));
        colorbar
        hold on
        set(gca , 'FontSize' , 16 ,'XTick' , [1:50] ,'YTick' , [1:length(interpSamp)] ,...
            'YTickLabel' , cellstr(num2str(interpSamp')));
        ylabel('Moving Window Size')
        title('Crossvalidated Loss of LDA on the interpolated moving window timeseries')
        finterpSamp = fliplr(interpSamp);
        for i = 1:length(finterpSamp)
            ind = round(linspace(1,50 , finterpSamp(i)));
            for j = 1:length(ind)
                line([ind(j)-.5 ind(j)-.5] , [i-.5 i+.5] ,'LineWidth',3,'LineStyle',':' , 'color' , 'm')
            end
        end
        
        line([16.5 16.5] , [.5 25.5] ,'LineWidth',3,'LineStyle',':' , 'color' , 'g')
        line([33.5 33.5] , [.5 25.5] ,'LineWidth',3,'LineStyle',':' , 'color' , 'g')
    case 'iterp_level_select'
        load([baseDir , '/Gest/Data_All.mat']);
        load([baseDir , '/Gest/CrossValidationIdx.mat'])
        % exclude the test fold
        D = getrow(Data , CVI ~= 11);
        
        
        switch Method
            case {'LDA'}
                for ip = 1:length(interpSamp)
                    for mn = 1:8
                        D = getrow(Data , CVI ~= 11);
                        for i = 1:length(D.GestNum)
                            D.interpEMG(i,:) = interp1([1:50] ,D.EMG{i}(:,mn), linspace(1,50 , interpSamp(ip)),'spline');
                            D.interpEMG(i,:) = D.interpEMG(i,:) - mean(D.interpEMG(i,:)); % normalizes each muscle
                        end
                        c = cvpartition(D.GestNum,'k',10);
                        Mdl = fitcdiscr(D.interpEMG , D.GestNum,'CV' , c);
                        loss(ip , mn) = kfoldLoss(Mdl);
                        disp(['Cross-Validated LDA - interpolation constant is  ' , num2str(interpSamp(ip)) , ', Muscle ' , num2str(mn) , ' completed!'])
                    end
                end
                
                [~ , id] = min(loss,[],1);
                bestInterp = interpSamp(id);
                save([baseDir , '/Gest/Crossvalidated_loss_LDA_interp_perMuscle.mat'] , 'loss' , 'interpSamp' , 'bestInterp')
                
                % pick the intepolation level by finding the min loss
                figure('color', 'white')
                imagesc(loss)
                colorbar
                hold on
                set(gca , 'FontSize' , 16 ,'XTick' , [1:8] , 'XTicklabel' , ...
                    {'Muscle 1' , 'Muscle 2' , 'Muscle 3' , 'Muscle 4' , 'Muscle 5' , 'Muscle 6' , 'Muscle 7' , 'Muscle 8'} , 'XTickLabelRotation' , 45,...
                    'YTick' , [1:length(interpSamp)] , 'YTickLabel' , cellstr(num2str(interpSamp')));
                ylabel('interpolation level')
                title('Crossvalidated Loss of LDA Classification')
                for i = 1:length(bestInterp)
                    x = i;
                    y = id(i);
                    rectangle('Position',[x - .5,y - .5,1,1],'LineWidth',3,'LineStyle',':' , 'EdgeColor' , 'm')
                end
            case {'logistic'}
                loss = zeros(length(interpSamp) , 8);
                for ip = 1:length(interpSamp)
                    for mn = 1:8
                        D = getrow(Data , CVI ~= 11);
                        for i = 1:length(D.GestNum)
                            D.interpEMG(i,:) = interp1([1:50] ,D.EMG{i}(:,mn), linspace(1,50 , interpSamp(ip)),'spline');
                        end
                        for cv= 1:10
                            id = CVI == cv; % test
                            X_train = D.interpEMG(~id , :);
                            X_test = D.interpEMG(id , :);
                            y_train  = D.GestNum(~id , :);
                            y_test  = D.GestNum(id , :);
                            [~ , falsePred] = Gest_fitlog(X_train,y_train,X_test,y_test);
                            loss(ip , mn) = loss(ip , mn) + falsePred/length(y_test);
                        end
                        disp(['Cross-Validated LDA - interpolation constant is  ' , num2str(interpSamp(ip)) , ', Muscle ' , num2str(mn) , ' completed!'])
                    end
                end
                loss = (loss/10);
                
                save([baseDir , '/Gest/Crossvalidated_loss_Log_interp_perMuscle.mat'] , 'loss' , 'interpSamp' , 'bestInterp')
                [~ , id] = min(loss,[],1);
                bestInterp = interpSamp(id);
                figure('color', 'white')
                imagesc(loss)
                colorbar
                hold on
                set(gca , 'FontSize' , 16 ,'XTick' , [1:8] , 'XTicklabel' , ...
                    {'Muscle 1' , 'Muscle 2' , 'Muscle 3' , 'Muscle 4' , 'Muscle 5' , 'Muscle 6' , 'Muscle 7' , 'Muscle 8'} , 'XTickLabelRotation' , 45,...
                    'YTick' , [1:length(interpSamp)] , 'YTickLabel' , cellstr(num2str(interpSamp')));
                ylabel('interpolation level')
                title('Crossvalidated Loss of Logistic Classification')
                for i = 1:length(bestInterp)
                    x = i;
                    y = id(i);
                    rectangle('Position',[x - .5,y - .5,1,1],'LineWidth',3,'LineStyle',':' , 'EdgeColor' , 'm')
                end
        end
        
        out.CVloss = loss;
        out.bestInterp_perchannel = bestInterp;
        out.interpSamp = interpSamp;
    case 'CrossVal_MuscleEnsemble'
        load([baseDir , '/Gest/Data_All.mat']);
        load([baseDir , '/Gest/CrossValidationIdx.mat'])
        load([baseDir , '/Gest/Crossvalidated_loss_LDA_interp_perMuscle.mat'])
        CVI_v = CVI( CVI ~= 11);
        switch Method
            case 'logistic'
                for cv= 1:10
                    clear pihat_weighted pihat
                    for mn = 1:8
                        D = getrow(Data , CVI ~= 11);
                        for i = 1:length(D.GestNum)
                            D.interpEMG(i,:) = interp1([1:50] ,D.EMG{i}(:,mn), linspace(1,50 , bestInterp(mn)),'spline');
                            D.interpEMG(i,:) - D.interpEMG(i,:) - mean(D.interpEMG(i,:));
                        end
                        id = CVI_v == cv; % test
                        X_train = D.interpEMG(~id , :);
                        X_test = D.interpEMG(id , :);
                        y_train  = D.GestNum(~id , :);
                        y_test  = D.GestNum(id , :);
                        
                        [pihat(mn ,: ,:) , ~] = Gest_fitlog(X_train,y_train,X_test,y_test);
                        disp(['Logistic cross-Validatedtion fold ', num2str(cv) , ', Muscle ' , num2str(mn) , ' completed!'])
                    end
                    CVProb{cv} = squeeze(median(pihat,1));
                    [~ , Gest]   = max(CVProb{cv} , [],2);
                    Conf{cv} = confusionmat(y_test,Gest);
                    Conf{cv} =  100*bsxfun(@rdivide, Conf{cv} , sum( Conf{cv} , 2));
                    Acc(cv) = sum(Gest==y_test)/length(y_test)
                end
                save([baseDir , '/Gest/Crossvalidated_Logistc_MuscleEnsemple.mat'] ,'Conf' , 'BW' , 'Acc' , 'Width');
                 Perf.X = [];
                Perf.Y = [];
                Perf.CV = [];
                Perf.y_test = [];
                Perf.postP = [];
                figure('color' , 'white')
                for g = 1:6
                    for cv= 1:10
                        id = CVI_v == cv;
                        y_test  = D.GestNum(id , :);
                        [X,Y] = perfcurve(y_test,max(CVProb{cv} , [],2),g);
                        Perf.X = [Perf.X ; X];
                        Perf.Y = [Perf.Y ; Y];
                        Perf.CV = [Perf.CV ; cv*ones(size(X))];
                        Perf.y_test = [Perf.y_test ; g*ones(size(X))];
                        hold on
                    end
                end
                figure('color' , 'white')
                for g = 1:6
                    subplot(2,3,g)
                    lineplot(Perf.X , Perf.Y  , 'subset' ,Perf.y_test==g );
                    set(gca , 'FontSize' , 16, 'XTick' , [0 .5 1], 'YTick' , [0 .5 1]);
                    title(['Crosvalidated logistic muscle Ensemble, gesture ' , num2str(g)])
                    ylabel('True Positive rate')
                    xlabel('False Positive rate')
                    hold on
                    line([0 1], [0 1], 'LineWidth' , 2,'LineStyle' , ':' , 'color' , 'r')
                    axis square
                end
                M = mean(cat(3,Conf{:}) , 3);
                figure('color' , 'white')
                imagesc(M)
                axis square
                xlabel('Gesture')
                ylabel('Gesture')
                set(gca , 'FontSize' , 16, 'XTick' , [1:6], 'YTick' , [1:6]);
                title('Crossvalidated Logistic  Muscle Ensemble')
                colorbar
                
            case 'NB'
                for cv= 1:10
                    clear pihat_weighted pihat
                    for mn = 1:8
                        D = getrow(Data , CVI ~= 11);
                        for i = 1:length(D.GestNum)
                            D.interpEMG(i,:) = interp1([1:50] ,D.EMG{i}(:,mn), linspace(1,50 , bestInterp(mn)),'spline');
                            D.interpEMG(i,:) - D.interpEMG(i,:) - mean(D.interpEMG(i,:));
                        end
                        id = CVI_v == cv; % test
                        X_train = D.interpEMG(~id , :);
                        X_test = D.interpEMG(id , :);
                        y_train  = D.GestNum(~id , :);
                        y_test  = D.GestNum(id , :);
                        [pihat(mn ,: ,:) , criterion , M(mn,cv).Mdl] = Gest_fitNaiveB(X_train,y_train,X_test,y_test);
                        disp(['NB cross-Validatedtion fold ', num2str(cv) , ', Muscle ' , num2str(mn) , ' completed!'])
                    end
                    CVProb{cv} = squeeze(median(pihat,1));
                    for g = 1:6
                        BW(cv ,mn,g) = M(mn,cv).Mdl.DistributionParameters{g}.BandWidth;
                    end
                    Width(cv , mn,:) = M(mn,cv).Mdl.Width;
                    [~ , Gest]   = max(CVProb{cv} , [],2);
                    Conf{cv} = confusionmat(y_test,Gest);
                    Conf{cv} =  100*bsxfun(@rdivide, Conf{cv} , sum( Conf{cv} , 2));
                    Acc(cv) = sum(Gest==y_test)/length(y_test)
                end
                save([baseDir , '/Gest/Crossvalidated_NaiveBayesClassifier_Distance.mat'] ,'Conf' , 'BW' , 'Acc' , 'Width');
               
                
                
        end
        out = [];
    case 'Interp_class_per_window_dist'
        load([baseDir , '/Gest/Data_All.mat']);
        load([baseDir , '/Gest/CrossValidationIdx.mat'])
        % exclude the test fold
        D = getrow(Data , CVI ~= 11);
        
        for ip = 1:length(interpSamp)
            ind = round(linspace(1,50 , interpSamp(ip)));
            for j = 1:length(ind)-1
                [ip j]
                D = getrow(Data , CVI ~= 11);
                for i = 1:length(D.GestNum)
                    D.interpDist(i , :) = pdist(transpose(D.EMG{i}(ind(j) : ind(j+1) , :)) , 'euclidean');
                end
                c = cvpartition(D.GestNum,'k',10);
                %                 F = Gest_applyPCA(D.interpDist  , D.GestNum , 15 , 0);
                Mdl = fitcdiscr(D.interpDist , D.GestNum,'CV' , c);
                loss(ip , j) = kfoldLoss(Mdl);
                
            end
            disp(['Cross-Validated LDA - window length is  ' , num2str(interpSamp(ip)) ,' completed!'])
        end
        
        save([baseDir , '/Gest/CrossValidated_MovingWindow_Distance_Loss.mat'] , 'L' , 'interpSamp')
        L = zeros(size(loss));
        for ip = 1:length(interpSamp)
            ind = round(linspace(1,50 , interpSamp(ip)));
            for j = 1:length(ind)-1
                L(ip , ind(j) : ind(j+1)-1) = repmat(loss(ip,j) , 1  , ind(j+1) - ind(j));
            end
        end
        figure('color', 'white')
        imagesc(flipud(L));
        colorbar
        hold on
        set(gca , 'FontSize' , 16 ,'XTick' , [1:50] ,'YTick' , [1:length(interpSamp)] ,...
            'YTickLabel' , cellstr(num2str(interpSamp')));
        ylabel('Moving Window Size')
        title('Crossvalidated Loss of LDA using distances ')
        finterpSamp = fliplr(interpSamp);
        for i = 1:length(finterpSamp)
            ind = round(linspace(1,50 , finterpSamp(i)));
            for j = 1:length(ind)
                line([ind(j)-.5 ind(j)-.5] , [i-.5 i+.5] ,'LineWidth',3,'LineStyle',':' , 'color' , 'm')
            end
        end
        
        line([16.5 16.5] , [.5 25.5] ,'LineWidth',3,'LineStyle',':' , 'color' , 'g')
        line([33.5 33.5] , [.5 25.5] ,'LineWidth',3,'LineStyle',':' , 'color' , 'g')
    case 'iterp_level_select_dist'
        load([baseDir , '/Gest/Data_All.mat']);
        load([baseDir , '/Gest/CrossValidationIdx.mat'])
        % exclude the test fold
        D = getrow(Data , CVI ~= 11);
        
        
        switch Method
            case {'LDA'}
                for ip = 1:length(interpSamp)
                    D = getrow(Data , CVI ~= 11);
                    for i = 1:length(D.GestNum)
                        D.interpEMG{i} = interp1([1:50] ,D.EMG{i}, linspace(1,50 , interpSamp(ip)),'spline');
                        D.interpDist(i , :) = pdist(transpose(D.interpEMG{i}) , 'euclidean');
                    end
                    c = cvpartition(D.GestNum,'k',10);
                    Mdl = fitcdiscr(D.interpDist , D.GestNum,'CV' , c);
                    loss(ip) = kfoldLoss(Mdl);
                    disp(['Cross-Validated LDA - interpolation constant is  ' , num2str(interpSamp(ip)) ,' completed!'])
                end
                
                id = find(diff(loss) > 0 , 1 , 'first');
                bestInterp = interpSamp(id);
                save([baseDir , '/Gest/Crossvalidated_loss_LDA_interp_Distance.mat'] , 'loss' , 'interpSamp' , 'bestInterp')
                
                
                % pick the intepolation level by finding the min loss
                figure('color', 'white')
                plot(loss,'-*','LineWidth',3 )
                grid on
                hold on
                set(gca , 'FontSize' , 16, 'XTick' , [1:length(interpSamp)] , 'XTickLabel' , cellstr(num2str(interpSamp')),'YLim' , [min(loss)-.02 max(loss)+.02]);
                ylabel('Loss')
                title('Crossvalidated Loss of LDA Classification on muscle pairwise distances')
                line([id id ] , [min(loss)-.02 max(loss)+.02] ,'LineWidth',3,'LineStyle',':' , 'color' , 'g')
                
        end
        
        out.CVloss = loss;
        out.bestInterp_perchannel = bestInterp;
        out.interpSamp = interpSamp;
    case 'FeatSelect'
        load([baseDir , '/Gest/Data_All.mat']);
        load([baseDir , '/Gest/CrossValidationIdx.mat'])
        load([baseDir , '/Gest/Crossvalidated_Weights_LassoReg_distance.mat'])
        
        
        D = getrow(Data , CVI ~= 11);
        CVI_v = CVI( CVI ~= 11);
        D = getrow(Data , CVI ~= 11);
        CVI_v = CVI( CVI ~= 11);
        for i = 1:length(D.GestNum)
            D.interpEMG{i,1} = interp1([1:50] ,D.EMG{i}, linspace(1,50 , bestInterp),'spline');
            D.interpDist(i , :) = pdist(transpose(D.interpEMG{i}) , 'euclidean');
        end
        out = Gest_featureSelect(D , 'NaiveB' , baseDir, 'interpDist');        
    case 'CrossVal_classify_dist'
        load([baseDir , '/Gest/Data_All.mat']);
        load([baseDir , '/Gest/CrossValidationIdx.mat'])
        load([baseDir , '/Gest/Crossvalidated_NB_Selected_Features_Distance.mat']);
        load([baseDir , '/Gest/Crossvalidated_loss_LDA_interp_Distance.mat']);
        
        if ~exist('AllFeatures')
            AllFeatures = 1;
        end
        if AllFeatures>=1
            Fs = 1:28;
        else
            Fs = find(fs);
        end
        D = getrow(Data , CVI ~= 11);
        CVI_v = CVI( CVI ~= 11);
        for i = 1:length(D.GestNum)
            D.interpEMG{i,1} = interp1([1:50] ,D.EMG{i}, linspace(1,50 , bestInterp),'spline');
            tempD = pdist(transpose(D.interpEMG{i}) , 'euclidean');
            D.interpDist(i , :) = tempD(Fs);
        end
        switch Method
            case 'logistic'
                for cv= 1:10
                    clear pihat_weighted pihat
                    id = CVI_v == cv;
                    
                    X_train = D.interpDist(~id , :);
                    X_test = D.interpDist(id , :);
                    y_train  = D.GestNum(~id , :);
                    y_test  = D.GestNum(id , :);
                    [pihat , ~ , B{cv}] = Gest_fitlog(X_train,y_train,X_test,y_test);
                    
                    disp(['Cross-Validatedtion fold ', num2str(cv) ,' completed!'])
                    
                    CVProb{cv} = pihat;
                    
                    [~ , Gest]   = max(CVProb{cv} , [],2);
                    Conf{cv} = confusionmat(y_test,Gest);
                    Conf{cv} =  100*bsxfun(@rdivide, Conf{cv} , sum( Conf{cv} , 2));
                    
                    Acc(cv) = sum(Gest==y_test)/length(y_test)
                    
                end
                Perf.X = [];
                Perf.Y = [];
                Perf.CV = [];
                Perf.y_test = [];
                Perf.postP = [];
                for g = 1:6
                    for cv= 1:10
                        id = CVI_v == cv;
                        y_test  = D.GestNum(id , :);
                        [X,Y] = perfcurve(y_test,max(CVProb{cv} , [],2),g);
                        Perf.X = [Perf.X ; X];
                        Perf.Y = [Perf.Y ; Y];
                        Perf.CV = [Perf.CV ; cv*ones(size(X))];
                        Perf.y_test = [Perf.y_test ; g*ones(size(X))];
                        hold on
                    end
                end
                figure('color' , 'white')
                for g = 1:6
                    subplot(2,3,g)
                    lineplot(Perf.X , Perf.Y  , 'subset' ,Perf.y_test==g );
                    set(gca , 'FontSize' , 16, 'XTick' , [0 .5 1], 'YTick' , [0 .5 1]);
                    title(['Crosvalidated ROC plot for gesture ' , num2str(g)])
                    ylabel('True Positive rate')
                    xlabel('False Positive rate')
                    hold on
                    line([0 1], [0 1], 'LineWidth' , 2,'LineStyle' , ':' , 'color' , 'r')
                    axis square
                end
                
                
                M = mean(cat(3,Conf{:}) , 3);
                figure('color' , 'white')
                imagesc(M)
                axis square
                xlabel('Gesture')
                ylabel('Gesture')
                set(gca , 'FontSize' , 16, 'XTick' , [1:6], 'YTick' , [1:6]);
                title('Crossvalidated Confusion Matrix - Logistic Classification on all the Distance on selected features')
                colorbar
                
                save([baseDir , '/Gest/Crossvalidated_LogisticClassifier_Distance_selectedFeat.mat'] , ...
                    'Conf' , 'Acc' , 'B', 'CVProb');
                out.B = B;
                out.Conf = Conf;
                out.Acc = Acc;
            case 'NB'
                for cv= 1:10
                    clear pihat_weighted pihat
                    id = CVI_v == cv;
                    
                    X_train = D.interpDist(~id , :);
                    X_test = D.interpDist(id , :);
                    y_train  = D.GestNum(~id , :);
                    y_test  = D.GestNum(id , :);
                    
                    %                     X_train = bsxfun(@minus, X_train , mean( X_train , 2));
                    %                     X_test = bsxfun(@minus, X_test , mean( X_test , 2));
                    
                    [pihat , criterion, M(cv).Mdl] = Gest_fitNaiveB(X_train,y_train,X_test,y_test , []);
                    
                    disp(['Cross-Validatedtion fold ', num2str(cv) ,' completed!'])
                    for g = 1:6
                        BW(cv,g) = M(cv).Mdl.DistributionParameters{g}.BandWidth;
                    end
                    Width(cv ,: ,:) = M(cv).Mdl.Width;
                    
                    CVProb{cv} = pihat;
                    
                    [~ , Gest]   = max(CVProb{cv} , [],2);
                    Conf{cv} = confusionmat(y_test,Gest);
                    Conf{cv} =  100*bsxfun(@rdivide, Conf{cv} , sum( Conf{cv} , 2));
                    
                    Acc(cv) = sum(Gest==y_test)/length(y_test)
                end
                save([baseDir , '/Gest/Crossvalidated_NaiveBayesClassifier_Distance.mat'] , ...
                    'Conf' , 'BW' , 'Acc' , 'Width' , 'CVProb');
                out.M = M;
                out.Conf = Conf;
                out.Acc = Acc;
                
                Perf.X = [];
                Perf.Y = [];
                Perf.CV = [];
                Perf.y_test = [];
                Perf.postP = [];
                for g = 1:6
                    for cv= 1:10
                        id = CVI_v == cv;
                        y_test  = D.GestNum(id , :);
                        [X,Y] = perfcurve(y_test,max(CVProb{cv} , [],2),g);
                        Perf.X = [Perf.X ; X];
                        Perf.Y = [Perf.Y ; Y];
                        Perf.CV = [Perf.CV ; cv*ones(size(X))];
                        Perf.y_test = [Perf.y_test ; g*ones(size(X))];
                        hold on
                    end
                end
                figure('color' , 'white')
                for g = 1:6
                    subplot(2,3,g)
                    lineplot(Perf.X , Perf.Y  , 'subset' ,Perf.y_test==g );
                    set(gca , 'FontSize' , 16, 'XTick' , [0 .5 1], 'YTick' , [0 .5 1]);
                    title(['Crosvalidated ROC plot for gesture ' , num2str(g)])
                    ylabel('True Positive rate')
                    xlabel('False Positive rate')
                    hold on
                    line([0 1], [0 1], 'LineWidth' , 2,'LineStyle' , ':' , 'color' , 'r')
                    axis square
                end
                
                
                M = mean(cat(3,Conf{:}) , 3);
                figure('color' , 'white')
                imagesc(M)
                axis square
                xlabel('Gesture')
                ylabel('Gesture')
                set(gca , 'FontSize' , 16, 'XTick' , [1:6], 'YTick' , [1:6]);
                title('Crossvalidated Confusion Matrix - NB Classification on all the Distances')
                colorbar
                
                
            case 'LDA'
                for cv= 1:10
                    clear pihat_weighted pihat
                    id = CVI_v == cv;
                    
                    X_train = D.interpDist(~id , :);
                    X_test = D.interpDist(id , :);
                    y_train  = D.GestNum(~id , :);
                    y_test  = D.GestNum(id , :);
                    
                    [pihat , criterion, M(cv).Mdl] = Gest_fitLDA(X_train,y_train,X_test,y_test);
                    
                    disp(['Cross-Validatedtion fold ', num2str(cv) ,' completed!'])
                    mu(cv ,: ,:) = M(cv).Mdl.Mu;
                    
                    CVProb{cv} = pihat;
                    
                    [~ , Gest]   = max(CVProb{cv} , [],2);
                    Conf{cv} = confusionmat(y_test,Gest);
                    Conf{cv} =  100*bsxfun(@rdivide, Conf{cv} , sum( Conf{cv} , 2));
                    
                    Acc(cv) = sum(Gest==y_test)/length(y_test);
                end
                save([baseDir , '/Gest/Crossvalidated_LDAClassifier_Distance.mat'] , ...
                    'Conf' , 'mu' , 'Acc');
                out.mu = mu;
                out.Conf = Conf;
                out.Acc = Acc;
                Perf.X = [];
                Perf.Y = [];
                Perf.CV = [];
                Perf.y_test = [];
                Perf.postP = [];
                for g = 1:6
                    for cv= 1:10
                        id = CVI_v == cv;
                        y_test  = D.GestNum(id , :);
                        [X,Y] = perfcurve(y_test,max(CVProb{cv} , [],2),g);
                        Perf.X = [Perf.X ; X];
                        Perf.Y = [Perf.Y ; Y];
                        Perf.CV = [Perf.CV ; cv*ones(size(X))];
                        Perf.y_test = [Perf.y_test ; g*ones(size(X))];
                        hold on
                    end
                end
                figure('color' , 'white')
                for g = 1:6
                    subplot(2,3,g)
                    lineplot(Perf.X , Perf.Y  , 'subset' ,Perf.y_test==g );
                    set(gca , 'FontSize' , 16, 'XTick' , [0 .5 1], 'YTick' , [0 .5 1]);
                    title(['Crosvalidated ROC plot for gesture ' , num2str(g)])
                    ylabel('True Positive rate')
                    xlabel('False Positive rate')
                    hold on
                    line([0 1], [0 1], 'LineWidth' , 2,'LineStyle' , ':' , 'color' , 'r')
                    axis square
                end
                
                
                M = mean(cat(3,Conf{:}) , 3);
                figure('color' , 'white')
                imagesc(M)
                axis square
                xlabel('Gesture')
                ylabel('Gesture')
                set(gca , 'FontSize' , 16, 'XTick' , [1:6], 'YTick' , [1:6]);
                title('Crossvalidated Confusion Matrix - LDA Classification on all the Distances')
                colorbar
            case 'SVM'
                for cv= 1:10
                    clear pihat_weighted pihat
                    id = CVI_v == cv;
                    
                    X_train = D.interpDist(~id , :);
                    X_test = D.interpDist(id , :);
                    y_train  = D.GestNum(~id , :);
                    y_test  = D.GestNum(id , :);
                    
                    [pihat , criterion, M(cv).Mdl] = Gest_fitSVM(X_train,y_train,X_test,y_test);
                    
                    disp(['Cross-Validatedtion fold ', num2str(cv) ,' completed!'])
                    
                    CVProb{cv} = pihat;
                    
                    [~ , Gest]   = max(CVProb{cv} , [],2);
                    Conf{cv} = confusionmat(y_test,Gest);
                    Conf{cv} =  100*bsxfun(@rdivide, Conf{cv} , sum( Conf{cv} , 2));
                    
                    Acc(cv) = sum(Gest==y_test)/length(y_test);
                end
                save([baseDir , '/Gest/Crossvalidated_SVMClassifier_Distance.mat'] , ...
                    'Conf' , 'M' , 'Acc');
                out.M = M;
                out.Conf = Conf;
                out.Acc = Acc;
        end
    case 'BuildAndTest_EndClassifier_dist'
        load([baseDir , '/Gest/Data_All.mat']);
        load([baseDir , '/Gest/CrossValidationIdx.mat'])
        load([baseDir , '/Gest/Crossvalidated_Weights_LassoReg_distance.mat'])
        
        
        D = Data;
        for i = 1:length(D.GestNum)
            D.interpEMG{i,1} = interp1([1:50] ,D.EMG{i}, linspace(1,50 , bestInterp),'spline');
            D.interpDist(i , :) = pdist(transpose(D.interpEMG{i}) , 'euclidean');
        end
        switch Method
            case 'logistic'
                id = CVI == 11;
                
                X_train = D.interpDist(~id , :);
                X_test = D.interpDist(id , :);
                y_train  = D.GestNum(~id , :);
                y_test  = D.GestNum(id , :);
                [pihat , ~ , B] = Gest_fitlog(X_train,y_train,X_test,y_test);
                
                disp(['Classifier completed!'])
                
                CVProb = pihat;
                
                [~ , Gest]   = max(CVProb , [],2);
                Conf = confusionmat(y_test,Gest);
                Conf =  100*bsxfun(@rdivide, Conf , sum( Conf , 2));
                
                Acc = sum(Gest==y_test)/length(y_test)
                
                
                figure('color' , 'white')
                subplot(122)
                hold on
                for g = 1:6
                    id = CVI == 11;
                    y_test  = D.GestNum(id , :);
                    [X,Y] = perfcurve(y_test,max(CVProb , [],2),g);
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
                
                
                
                
                M = Conf;
                subplot(121)
                imagesc(M)
                axis square
                xlabel('Gesture')
                ylabel('Gesture')
                set(gca , 'FontSize' , 16, 'XTick' , [1:6], 'YTick' , [1:6]);
                title('Test Confusion Matrix - Logistic Classification on all the Distance')
                colorbar
                
                save([baseDir , '/Gest/EndClassifiers/END_LogisticClassifier_Distance.mat'] , ...
                    'Conf' , 'Acc' , 'B', 'CVProb');
                out.B = B;
                out.Conf = Conf;
                out.Acc = Acc;
            case 'logistic_Sel'
                load([baseDir , '/Gest/Crossvalidated_NB_Selected_Features_Distance.mat']);
                id = CVI == 11;
                
                X_train = D.interpDist(~id , find(fs));
                X_test = D.interpDist(id , find(fs));
                y_train  = D.GestNum(~id , :);
                y_test  = D.GestNum(id , :);
                [pihat , ~ , B] = Gest_fitlog(X_train,y_train,X_test,y_test);
                
                disp(['Classifier completed!'])
                
                CVProb = pihat;
                
                [~ , Gest]   = max(CVProb , [],2);
                Conf = confusionmat(y_test,Gest);
                Conf =  100*bsxfun(@rdivide, Conf , sum( Conf , 2));
                
                Acc = sum(Gest==y_test)/length(y_test)
                
                figure('color' , 'white')
                subplot(122)
                hold on
                for g = 1:6
                    id = CVI == 11;
                    y_test  = D.GestNum(id , :);
                    [X,Y] = perfcurve(y_test,max(CVProb , [],2),g);
                    plot(X , Y, 'LineWidth' , 3 );
                    grid on
                    set(gca , 'FontSize' , 16, 'XTick' , [0 .5 1], 'YTick' , [0 .5 1]);
                    title(['Test ROC plot Logistic Regression - Selected distances'])
                    ylabel('True Positive rate')
                    xlabel('False Positive rate')
                    hold on
                    axis square
                end
                legend({'G1' , 'G2' , 'G3' , 'G4' , 'G5' , 'G6'})
                line([0 1], [0 1], 'LineWidth' , 2,'LineStyle' , ':' , 'color' , 'r')
                
                
                
                
                M = Conf;
                subplot(121)
                imagesc(M)
                axis square
                xlabel('Gesture')
                ylabel('Gesture')
                set(gca , 'FontSize' , 16, 'XTick' , [1:6], 'YTick' , [1:6]);
                title('Test Confusion Matrix - Logistic Classification on selected the Distance')
                colorbar
                
                save([baseDir , '/Gest/EndClassifiers/END_LogisticClassifier_SelectedDistance.mat'] , ...
                    'Conf' , 'Acc' , 'B', 'CVProb');
                out.B = B;
                out.Conf = Conf;
                out.Acc = Acc;
            case 'NB'
                id = CVI == 11;
                
                X_train = D.interpDist(~id , :);
                X_test = D.interpDist(id , :);
                y_train  = D.GestNum(~id , :);
                y_test  = D.GestNum(id , :);
                
                [pihat , criterion, Mdl] = Gest_fitNaiveB(X_train,y_train,X_test,y_test , []);
                disp(['Classifier completed!'])
                
                CVProb = pihat;
                
                [~ , Gest]   = max(CVProb , [],2);
                Conf = confusionmat(y_test,Gest);
                Conf =  100*bsxfun(@rdivide, Conf , sum( Conf , 2));
                
                Acc = sum(Gest==y_test)/length(y_test)
                
                
                figure('color' , 'white')
                subplot(122)
                hold on
                for g = 1:6
                    id = CVI == 11;
                    y_test  = D.GestNum(id , :);
                    [X,Y] = perfcurve(y_test,max(CVProb , [],2),g);
                    plot(X , Y, 'LineWidth' , 3 );
                    grid on
                    set(gca , 'FontSize' , 16, 'XTick' , [0 .5 1], 'YTick' , [0 .5 1]);
                    title(['Test ROC plot - Naive Bayes'])
                    ylabel('True Positive rate')
                    xlabel('False Positive rate')
                    hold on
                    axis square
                end
                legend({'G1' , 'G2' , 'G3' , 'G4' , 'G5' , 'G6'})
                line([0 1], [0 1], 'LineWidth' , 2,'LineStyle' , ':' , 'color' , 'r')
                
                
                
                
                M = Conf;
                subplot(121)
                imagesc(M)
                axis square
                xlabel('Gesture')
                ylabel('Gesture')
                set(gca , 'FontSize' , 16, 'XTick' , [1:6], 'YTick' , [1:6]);
                title('Test Confusion Matrix - Naive Bayes Classification on all the Distance')
                colorbar
                
                
                
                save([baseDir , '/Gest/EndClassifiers/END_NaiveBayesClassifier_Distance.mat'] , ...
                    'Conf' , 'Acc' , 'Mdl' , 'CVProb');
                out.M = M;
                out.Conf = Conf;
                out.Acc = Acc;
                
            case 'LDA'
                id = CVI == 11;
                
                X_train = D.interpDist(~id , :);
                X_test = D.interpDist(id , :);
                y_train  = D.GestNum(~id , :);
                y_test  = D.GestNum(id , :);
                
                
                [pihat , criterion, Mdl] = Gest_fitLDA(X_train,y_train,X_test,y_test);
                
                disp(['Classifier completed!'])
                
                CVProb = pihat;
                
                [~ , Gest]   = max(CVProb , [],2);
                Conf = confusionmat(y_test,Gest);
                Conf =  100*bsxfun(@rdivide, Conf , sum( Conf , 2));
                
                Acc = sum(Gest==y_test)/length(y_test)
                
                
                figure('color' , 'white')
                subplot(122)
                hold on
                for g = 1:6
                    id = CVI == 11;
                    y_test  = D.GestNum(id , :);
                    [X,Y] = perfcurve(y_test,max(CVProb , [],2),g);
                    plot(X , Y, 'LineWidth' , 3 );
                    grid on
                    set(gca , 'FontSize' , 16, 'XTick' , [0 .5 1], 'YTick' , [0 .5 1]);
                    title(['Test ROC plot - LDA'])
                    ylabel('True Positive rate')
                    xlabel('False Positive rate')
                    hold on
                    axis square
                end
                legend({'G1' , 'G2' , 'G3' , 'G4' , 'G5' , 'G6'})
                line([0 1], [0 1], 'LineWidth' , 2,'LineStyle' , ':' , 'color' , 'r')
                
                
                
                
                M = Conf;
                subplot(121)
                imagesc(M)
                axis square
                xlabel('Gesture')
                ylabel('Gesture')
                set(gca , 'FontSize' , 16, 'XTick' , [1:6], 'YTick' , [1:6]);
                title('Test Confusion Matrix - LDA Classification on all the Distance')
                colorbar
                
                
                save([baseDir , '/Gest/EndClassifiers/END_LDAClassifier_Distance.mat'] , ...
                    'Conf' , 'Mdl' , 'Acc');
                out.Mdl = Mdl;
                out.Conf = Conf;
                out.Acc = Acc;
            
        end
end


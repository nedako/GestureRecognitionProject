function out = Gest_analyze(what , baseDir , varargin)


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
        otherwise
            error(sprintf('No such option: %s',varargin{c}));
    end
    
end
if ~exist('whichInterp')
    whichInterp = 7;      % posthoc - cz I know!
end

switch what
    case 'Unpack'
        % unpack the EMG data and amend the data structure with summaries of the data and save to disc
        load([baseDir , '/Data_All.mat']);
        Temp.EMG      = cell2mat(Data.EMG);
        Temp.timeStapm = repmat([1:50]' , length(Data.EMG) , 1);
        Temp.TiralNum = repmat(Data.TiralNum , 50 , 1);
        Temp.GestNum  = repmat(Data.GestNum , 50 , 1);
        % Summerize the data by acquiring the median muscular activity for each trial (collaps over time dimension)
        All = tapply(Temp , {'TiralNum' , 'GestNum'} , {'EMG' , 'nanmedian(x)' , 'name' , 'medianEMG'});
        
        
        Data.medianEMG = All.medianEMG;
        
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
        save([baseDir , '/CrossValidationIdx.mat'] , 'CVI')
        save([baseDir , '/' , saveName] , 'Data')
        
        out = Data;
    case 'VisualizeSummary'
        load([baseDir , '/Data_All.mat']);
        colorz = {[0 0  1],[1 0 0],[0 1 0],[1 0 1],[0 1 1],[0.7 0.7 0.7],[1 1 0],[.3 .3 .3]};
        clear temp
        Temp.EMG      = cell2mat(Data.EMG);
        Temp.TiralNum = repmat(Data.TiralNum , 50 , 1);
        Temp.GestNum  = repmat(Data.GestNum , 50 , 1);
        Temp.timeStapm = repmat([1:50]' , length(Data.EMG) , 1);
        
        %========= create plotting indecies for interpolated data of whichInterp
        eval(['Temp.interpEMG      = cell2mat(Data.interpEMG',num2str(whichInterp),');']);
        Temp.interpTime  = repmat(linspace(1,50 , whichInterp)' , length(Data.EMG) , 1);
        Temp.iGestNum  = repmat(Data.GestNum , whichInterp , 1);
        Temp.iTiralNum = repmat(Data.TiralNum , whichInterp , 1);
        
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
        load([baseDir , '/Data_All.mat']);
        load([baseDir , '/CrossValidationIdx.mat'])
        % exclude the test fold
        D = getrow(Data , CVI ~= 11);
        
        for ip = 1:length(interpSamp)
            for i = 1:length(D.GestNum)
                D.interpEMG{i,1} = interp1([1:50] ,D.EMG{i}, linspace(1,50 , interpSamp(ip)),'spline');
            end
            Temp.interpEMG      = cell2mat(D.interpEMG);
            Temp.interpTime  = repmat(linspace(1,50 , interpSamp(ip))' , length(D.EMG) , 1);
            Temp.iGestNum  = repmat(D.GestNum , interpSamp(ip) , 1);
            
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
        save([baseDir , '/CrossValidated_MovingWindow_Loss.mat'] , 'L' , 'interpSamp')
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
        load([baseDir , '/Data_All.mat']);
        load([baseDir , '/CrossValidationIdx.mat'])
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
                save([baseDir , '/Crossvalidated_loss_LDA_interp_perMuscle.mat'] , 'loss' , 'interpSamp' , 'bestInterp')
                
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
                
                save([baseDir , '/Crossvalidated_loss_Log_interp_perMuscle.mat'] , 'loss' , 'interpSamp' , 'bestInterp')
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
    case 'Elastic_Weight'
        load([baseDir , '/Data_All.mat']);
        load([baseDir , '/CrossValidationIdx.mat'])
        load([baseDir , '/Crossvalidated_loss_LDA_interp_perMuscle.mat'])
        % exclude the test fold
        
        Data.X = [zeros(length(Data.TiralNum) ,6)];
        for gn = 1:6
            id = Data.GestNum == gn;
            eval(['Data.X(id ,', num2str(gn) ') = 1;']);
        end
        
        for mn = 1:8
            % exclude the test fold
            D = getrow(Data , CVI ~= 11);
            for i = 1:length(D.GestNum)
                D.interpEMG(i,:) = interp1([1:50] ,D.EMG{i}(:,mn), linspace(1,50 , bestInterp(mn)),'spline');
                D.interpEMG(i,:) = D.interpEMG(i,:) - mean(D.interpEMG(i,:));
            end
            c = cvpartition(D.GestNum,'k',10);
            [BLasso{mn} MdlInf(mn).M] = lassoglm(D.X , D.interpEMG,'normal','CV' , c,'Alpha' , .8 , 'lambda' , [0.1:0.1:.9]);
            Weights(mn , :) = BLasso{mn}(: , MdlInf(mn).M.IndexMinDeviance);
            Lambda(mn , :) = MdlInf(mn).M.LambdaMinDeviance;
        end
        out.Weights = Weights;
        
        save([baseDir , '/Crossvalidated_Weights_LassoReg.mat'] , 'Weights' , 'Lambda' , 'bestInterp');
    case 'Ensemble_Boost'
        load([baseDir , '/Data_All.mat']);
        load([baseDir , '/CrossValidationIdx.mat'])
        load([baseDir , '/Crossvalidated_Weights_LassoReg.mat'])
        
        
        for cv= 1:10
            clear pihat_weighted pihat
            for mn = 1:8
                D = getrow(Data , CVI ~= 11);
                for i = 1:length(D.GestNum)
                    D.interpEMG(i,:) = interp1([1:50] ,D.EMG{i}(:,mn), linspace(1,50 , bestInterp(mn)),'spline');
                    D.interpEMG(i,:) - D.interpEMG(i,:) - mean(D.interpEMG(i,:));
                end
                id = CVI == cv; % test
                X_train = D.interpEMG(~id , mn);
                X_test = D.interpEMG(id , mn);
                y_train  = D.GestNum(~id , :);
                y_test  = D.GestNum(id , :);
                %                 [pihat(mn ,: ,:) , ~] = Gest_fitlog(X_train,y_train,X_test,y_test);
                [pihat(mn ,: ,:) , criterion , M(mn,cv).Mdl] = Gest_fitNaiveB(X_train,y_train,X_test,y_test);
                w =  Weights(mn , :);
                w(w <= 0) = 1;
                pihat_weighted(mn , :,:) = bsxfun(@times,squeeze(pihat(mn ,:,:)),w);
                disp(['Cross-Validatedtion fold ', num2str(cv) , ', Muscle ' , num2str(mn) , ' completed!'])
            end
            
            CVProb{cv} = squeeze(median(pihat,1));
            CVProb_weighted{cv} = squeeze(median(pihat_weighted,1));
            
            [~ , Gest]   = max(CVProb{cv} , [],2);
            [~ , Gest_weighted]   = max(CVProb_weighted{cv} , [],2);
            Conf{cv} = confusionmat(y_test,Gest);
            Conf{cv} =  100*bsxfun(@rdivide, Conf{cv} , sum( Conf{cv} , 2));
            Conf_weighted{cv} = confusionmat(y_test,Gest_weighted);
            Conf_weighted{cv} =  100*bsxfun(@rdivide, Conf_weighted{cv} , sum( Conf_weighted{cv} , 2));
            Acc(cv) = sum(Gest==y_test)/length(y_test)
            Acc_weighted(cv) = sum(Gest_weighted==y_test)/length(y_test)
        end
        out.M = M;
        out.Conf = Conf;
        out.Conf_weighted = Conf_weighted;
        out.Acc = Acc;
        out.Acc_weighted = Acc_weighted;
        
    case 'Interp_class_per_window_dist'
        load([baseDir , '/Data_All.mat']);
        load([baseDir , '/CrossValidationIdx.mat'])
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
        L = zeros(size(loss));
        for ip = 1:length(interpSamp)
            ind = round(linspace(1,50 , interpSamp(ip)));
            for j = 1:length(ind)-1
                L(ip , ind(j) : ind(j+1)-1) = repmat(loss(ip,j) , 1  , ind(j+1) - ind(j));
            end
        end
        save([baseDir , '/CrossValidated_MovingWindow_Distance_Loss.mat'] , 'L' , 'interpSamp')
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
        load([baseDir , '/Data_All.mat']);
        load([baseDir , '/CrossValidationIdx.mat'])
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
                save([baseDir , '/Crossvalidated_loss_LDA_interp_Distance.mat'] , 'loss' , 'interpSamp' , 'bestInterp')
                
                
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
    case 'Elastic_Weight_dist'
        load([baseDir , '/Data_All.mat']);
        load([baseDir , '/CrossValidationIdx.mat'])
        load([baseDir , '/Crossvalidated_loss_LDA_interp_Distance.mat'])
        % exclude the test fold
        
        Data.X = [zeros(length(Data.TiralNum) ,6)];
        for gn = 1:6
            id = Data.GestNum == gn;
            eval(['Data.X(id ,', num2str(gn) ') = 1;']);
        end
        
        
        D = getrow(Data , CVI ~= 11);
        for i = 1:length(D.GestNum)
            D.interpEMG{i,1} = interp1([1:50] ,D.EMG{i}, linspace(1,50 , bestInterp),'spline');
            D.interpDist(i , :) = pdist(transpose(D.interpEMG{i}) , 'euclidean');
        end
        c = cvpartition(D.GestNum,'k',10);
        for d = 1:size(D.interpDist , 2)
            [BLasso{d} MdlInf(d).M] = lassoglm(D.X , D.interpDist(:,d),'normal','CV' , c,'Alpha' , 0.01 , 'lambda' , [0.1:0.1:.9]);
            Weights(d , :) = BLasso{d}(: , MdlInf(d).M.IndexMinDeviance);
            Lambda(d , :) = MdlInf(d).M.LambdaMinDeviance;
            disp(['Distance ' , num2str(d)  , ' of 28 completed!'])
        end
        
        save([baseDir , '/Crossvalidated_Weights_LassoReg_distance.mat'] , 'Weights' , 'Lambda' , 'bestInterp');
        out.Weights = Weights;
    case 'Ensemble_Boost_dist'
        load([baseDir , '/Data_All.mat']);
        load([baseDir , '/CrossValidationIdx.mat'])
        load([baseDir , '/Crossvalidated_Weights_LassoReg_distance.mat'])
        
        
        D = getrow(Data , CVI ~= 11);
        CVI_v = CVI( CVI ~= 11);
        for i = 1:length(D.GestNum)
            D.interpEMG{i,1} = interp1([1:50] ,D.EMG{i}, linspace(1,50 , bestInterp),'spline');
            D.interpDist(i , :) = pdist(transpose(D.interpEMG{i}) , 'euclidean');
        end
        c = cvpartition(D.GestNum,'k',10);
        
        for cv= 1:10
            clear pihat_weighted pihat
            id = CVI_v == cv;
            for d = 1:min(size(D.interpDist))
                X_train = D.interpDist(~id , d);
                X_test = D.interpDist(id , d);
                y_train  = D.GestNum(~id , :);
                y_test  = D.GestNum(id , :);
%                 [pihat(d ,: ,:) , ~] = Gest_fitlog(X_train,y_train,X_test,y_test);
                [pihat(d ,: ,:) , ~, Mdl] = Gest_fitSVM(X_train,y_train,X_test,y_test);
%                 [pihat(d ,: ,:) , criterion, M(d,cv).Mdl] = Gest_fitNaiveB(X_train,y_train,X_test,y_test);
                w =  Weights(d , :);
                w(w<0) = 1;
                pihat_weighted(d , :,:) = bsxfun(@times,squeeze(pihat(d ,:,:)),w);
                disp(['Cross-Validatedtion fold ', num2str(cv) , ', distance ' , num2str(d) , ' completed!'])
%                 for g = 1:6
%                     BW(cv ,d,g) = M(d,cv).Mdl.DistributionParameters{g}.BandWidth;
%                 end
%                 Width(cv , d,:) = M(d,cv).Mdl.Width;
            end
            
            % Sum 3D matrix along 3rd dimension
            CVProb{cv} = squeeze(median(pihat,1));
            CVProb_weighted{cv} = squeeze(median(pihat_weighted,1));
            
            [~ , Gest]   = max(CVProb{cv} , [],2);
            [~ , Gest_weighted]   = max(CVProb_weighted{cv} , [],2);
            Conf{cv} = confusionmat(y_test,Gest);
            Conf{cv} =  100*bsxfun(@rdivide, Conf{cv} , sum( Conf{cv} , 2));
            Conf_weighted{cv} = confusionmat(y_test,Gest_weighted);
            Conf_weighted{cv} =  100*bsxfun(@rdivide, Conf_weighted{cv} , sum( Conf_weighted{cv} , 2));
            Acc(cv) = sum(Gest==y_test)/length(y_test)
            Acc_weighted(cv) = sum(Gest_weighted==y_test)/length(y_test)
        end  
        save([baseDir , '/Crossvalidated_NaiveBayesClassifier_Distance.mat'] , ...
            'Conf' , 'Conf_weighted' , 'Acc' , 'Acc_weighted');
        out.M = M;
        out.Conf = Conf;
        out.Conf_weighted = Conf_weighted;
        out.Acc = Acc;
        out.Acc_weighted = Acc_weighted;
        
    case 'classify_dist'
        load([baseDir , '/Data_All.mat']);
        load([baseDir , '/CrossValidationIdx.mat'])
        load([baseDir , '/Crossvalidated_Weights_LassoReg_distance.mat'])
        
        
        D = getrow(Data , CVI ~= 11);
        CVI_v = CVI( CVI ~= 11);
        for i = 1:length(D.GestNum)
            D.interpEMG{i,1} = interp1([1:50] ,D.EMG{i}, linspace(1,50 , bestInterp),'spline');
            D.interpDist(i , :) = pdist(transpose(D.interpEMG{i}) , 'euclidean');
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
                save([baseDir , '/Crossvalidated_LogisticClassifier_Distance.mat'] , ...
                    'Conf' , 'Acc' , 'B');
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
                save([baseDir , '/Crossvalidated_NaiveBayesClassifier_Distance.mat'] , ...
                    'Conf' , 'BW' , 'Acc' , 'Width');
                out.M = M;
                out.Conf = Conf;
                out.Acc = Acc;
                
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
                save([baseDir , '/Crossvalidated_LDAClassifier_Distance.mat'] , ...
                    'Conf' , 'mu' , 'Acc');
                out.mu = mu;
                out.Conf = Conf;
                out.Acc = Acc;
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
                save([baseDir , '/Crossvalidated_SVMClassifier_Distance.mat'] , ...
                    'Conf' , 'M' , 'Acc');
                out.M = M;
                out.Conf = Conf;
                out.Acc = Acc;
        end
    case 'FeatSelect'
        load([baseDir , '/Data_All.mat']);
        load([baseDir , '/CrossValidationIdx.mat'])
        load([baseDir , '/Crossvalidated_Weights_LassoReg_distance.mat'])
        
        
        D = getrow(Data , CVI ~= 11);
        CVI_v = CVI( CVI ~= 11);
        D = getrow(Data , CVI ~= 11);
        CVI_v = CVI( CVI ~= 11);
        for i = 1:length(D.GestNum)
            D.interpEMG{i,1} = interp1([1:50] ,D.EMG{i}, linspace(1,50 , bestInterp),'spline');
            D.interpDist(i , :) = pdist(transpose(D.interpEMG{i}) , 'euclidean');
        end
        out = Gest_featureSelect(D , 'NaiveB' , baseDir, 'interpDist');
end


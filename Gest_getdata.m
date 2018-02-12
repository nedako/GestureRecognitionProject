function [Data ,isBad] = Gest_getdata(baseDir , what , varargin)
%%
% Reads and saves data to disc
%   [Data ,isBad] = Gest_importData(baseDir , what , varargin)
% Description
%   baseDir : where the data is saved
%   what: data reading options
% 
%       'All'     : read all the data save in baseDir
%       'Single'  : read a single trial - specify name
%       'Gesture' : read all the trials from a certian gesture - specify number
%   varargin: options
%       'savedata'  : 1 or 0 -  to save or not save the data to disc
%       'trialfile' : for what = 'single' --> name of the single trial file to be read             
%       'gestureNum': for what = 'Gesture' --> number of the spcific gesture to be read
% 
% 
% Neda Kordjazi 
% November 2017
%%


tic
Data.EMG      = {};
Data.TiralNum = [];
Data.GestNum  = [];

isBad.TiralNum = [];
isBad.GestNum   = [];

cd(baseDir);
DirCont = dir;
c = 1;
while(c<=length(varargin))
    switch(varargin{c})
        case {'savedata'}
            eval([varargin{c} '= (varargin{c+1} >= 1);']);
            c=c+2;
        case {'trialfile'}
            eval([varargin{c} '= varargin{c+1};']);
            c=c+2;
            if ~strcmp(what , 'Single')
                disp('You entered a trial name. Only that trial will be read.')
                what = 'Single';
            end
        case {'gesturenum'}
            eval([varargin{c} '= varargin{c+1};']);
            if varargin{c+1}>6 | varargin{c+1}<1
                error('Gesture Number should be between 1 - 6')
            else
                
                if ~strcmp(what , 'Gesture')
                    disp('You entered a Gesture number. Only the trials for that Gesture will be read.')
                    what = 'Gesture';
                end
            end
            c=c+2;
        otherwise
            error(sprintf('Unknown option: %s',varargin{c}));
    end
    
end
if ~exist('savedata')
    savedata = 1;
end
switch what
    case 'All'
        wb = waitbar(0,'Reading trials...');
        trls = length(DirCont);
        for i = 1:length(DirCont)
            if length(DirCont(i).name)>=17 & strcmp(DirCont(i).name(1:7), 'Gesture')
                filename      = DirCont(i).name;
                emg = csvread(filename);
                if max(size(emg)) ~= 50 | min(size(emg)) ~= 8
                    isBad.TiralNum = [isBad.TiralNum ; str2num(filename(17:end-4))];
                    isBad.GestNum   = [isBad.GestNum ; str2num(filename(8))];
                else
                    if size(emg , 1) == 8
                        emg = emg';
                    end
                    Data.EMG      = [Data.EMG      ; {emg}];
                    Data.TiralNum = [Data.TiralNum ; str2num(filename(17:end-4))];
                    Data.GestNum  = [Data.GestNum  ; str2num(filename(8))];
                end
            end
            waitbar(i/trls)
        end
        close(wb)
        disp(['Reading all the trails took ', num2str(toc) , ' Seconds'])
        if savedata
            save([baseDir ,'/Gest/Data_All.mat'] , 'Data' , '-v7.3');
            save([baseDir ,'/Gest/isBad_All.mat'] , 'isBad');
            disp('Saving...')
            disp('The Data and isBad structures are saved to the baseDir directory')
        end
    case 'Single'
        if ~exist('trialfile')
            error(sprintf('No trial name provided.'));
        end
        filename      = trialfile;
        if length(filename)>=17 & strcmp(filename(1:7), 'Gesture')
            
            emg = csvread(filename);
            if max(size(emg)) ~= 50 | min(size(emg)) ~= 8
                isBad.TiralNum = [isBad.TiralNum ; str2num(filename(17:end-4))];
                isBad.GestNum   = [isBad.GestNum ; str2num(filename(8))];
            else
                if size(emg , 1) == 8
                    emg = emg';
                end
                Data.EMG      = [Data.EMG      ; {emg}];
                Data.TiralNum = [Data.TiralNum ; str2num(filename(17:end-4))];
                Data.GestNum  = [Data.GestNum  ; str2num(filename(8))];
            end
        end
        if savedata
            save([baseDir ,'/Gest/',filename(1:end-4) , '.mat'] , 'Data');
            save([baseDir ,'/Gest/isBad_' , filename(1:end-4) , '.mat'] , 'isBad');
            disp('Saving...')
            disp('The Data and isBad structures are saved to the baseDir directory')
        end
    case 'Gesture'
        if ~exist('gestureNum')
            error(sprintf('No gesture number name provided.'));
        end
        wanted = [];
        
        for i = 1:length(DirCont)
            if length(DirCont(i).name)>=17 & strcmp(DirCont(i).name(1:8), ['Gesture' , num2str(gestureNum)])
                wanted = [wanted ; i];
            end
        end
        wb = waitbar(0,'Reading trials...');
        trls = length(wanted);
        for j = 1:length(wanted)
            i = wanted(j);
            filename      = DirCont(i).name;
            emg = csvread(filename);
            if max(size(emg)) ~= 50 | min(size(emg)) ~= 8
                isBad.TiralNum = [isBad.TiralNum ; str2num(filename(17:end-4))];
                isBad.GestNum   = [isBad.GestNum ; str2num(filename(8))];
            else
                if size(emg , 1) == 8
                    emg = emg';
                end
                Data.EMG      = [Data.EMG      ; {emg}];
                Data.TiralNum = [Data.TiralNum ; str2num(filename(17:end-4))];
                Data.GestNum  = [Data.GestNum  ; str2num(filename(8))];
            end
            waitbar(j/trls)
        end
       close(wb)
       disp(['Reading the trails took ', num2str(toc) , ' Seconds'])
        if savedata
            save([baseDir ,'/Gest/Data_G',num2str(gestureNum),'.mat'] , 'Data' , '-v7.3');
            save([baseDir ,'/Gest/isBad_All_G',num2str(gestureNum),'.mat'] , 'isBad');
            disp('Saving...')
            disp('The Data and isBad structures are saved to the baseDir directory')
        end
end



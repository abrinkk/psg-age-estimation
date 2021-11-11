function [ar,ar_seq] = LoadAR(p_file,L,ftype)
%LOADAR reads arousal annotation files.
%   [ar,ar_seq] = LOADAR(p_file,L,ftype) loads the annotation file
%   specified in p_file and processes the annotatinos to match function
%   output.
%
%   Author: Andreas Brink-Kjaer.
%   Date: 02-Feb-2021
%
%   Input:  p_file, annotation file location
%           L, annotation length (should be based on PSG length
%           ftype, string of data source
%   Output: ar, arousal data structure
%           ar_seq, arousal labels in 1 second bins

% Determine data type
if ~exist('ftype','var')
    if contains(p_file,'mros')
        ftype = 'mros';
    elseif contains(p_file,'cfs')
        ftype = 'cfs';
    end
end

switch ftype
    case {'cfs', 'mros','shhs'}
        % Read XML file
        s = xml2struct(p_file);
        events = s.CMPStudyConfig.ScoredEvents.ScoredEvent;
        ar = struct;
        ar_seq = zeros(1,L);
        k = 1;
        % Iterate over all events and save arousals
        for i = 1:length(events)
            if contains(events{i}.Name.Text,'Arousal')
                ar.start(k) = str2num(events{i}.Start.Text);
                ar.duration(k) = str2num(events{i}.Duration.Text);
                ar.stop(k) = ar.start(k) + ar.duration(k);
                ar_seq(ceil(ar.start(k)):ceil(ar.stop(k))) = 1;
                k = k + 1;
            end
        end
        ar_seq = ar_seq(1:L);
    case 'wsc2'
        % Read csv annotation file
        T = readtable(p_file,'FileType','text','Delimiter',';');
        T = T(:,end-1:end);
        T.Properties.VariableNames = {'Time','Event'};
        T(cellfun(@isempty, T.Time),:) = [];
        % Get event time in seconds
        time_all = time2ind(T);
        time_ar = time_all(contains(T.Event,'Arousal'));
        ar = struct;
        ar_seq = zeros(1,L);
        % Iterate and save all arousals
        for i = 1:length(time_ar)
            ar.start(i) = time_ar(i);
            ar.duration(i) = 3;
            ar.stop(i) = ar.start(i) + ar.duration(i);
            ar_seq(ceil(ar.start(i)):ceil(ar.stop(i))) = 1;
        end
        ar_seq = ar_seq(1:L);
    case 'ssc'
        % Read .EVTS file
%         [SCO,~] = CLASS_codec.parseSSCevtsFile(p_file);
%         ar_idx = contains(SCO.category,'arousal');
%         ar_event = SCO.startStopSamples(ar_idx,:)/SCO.samplerate;
        ar = struct;
        ar_seq = zeros(1,L);
        % Iterate over all arousals and save them
%         for i = 1:size(ar_event,1)
%             ar.start(i) = ar_event(i,1);
%             ar.duration(i) = ar_event(i,2)-ar_event(i,1);
%             ar.stop(i) = ar.start(i) + ar.duration(i);
%             ar_seq(ceil(ar.start(i)):ceil(ar.stop(i))) = 1;
%         end
%         ar_seq = ar_seq(1:L);
    case 'wsc'
        % Return empty labels (none exists)
        ar = struct;
        ar_seq = zeros(1,L);
    case 'stages'
        ar = struct;
        ar_seq = zeros(1,L);
end

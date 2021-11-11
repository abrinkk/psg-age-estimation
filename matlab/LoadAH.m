function [ah,ah_seq] = LoadAH(p_file,L,ftype)
%LOADAR reads arousal annotation files.
%   [ah,ah_seq] = LOADAH(p_file,L,ftype) loads the annotation file
%   specified in p_file and processes the annotatinos to match function
%   output.
%
%   Author: Andreas Brink-Kjaer.
%   Date: 02-Feb-2021
%
%   Input:  p_file, annotation file location
%           L, annotation length (should be based on PSG length
%           ftype, string of data source
%   Output: ah, apnea-hyponea data structure
%           ah_seq, apnea-hyponea labels in 1 second bins

% Determine data type
if ~exist('ftype','var')
    if contains(p_file,'mros')
        ftype = 'mros';
    elseif contains(p_file,'cfs')
        ftype = 'cfs';
    elseif contains(p_file,'shhs')
        ftype = 'shhs';
    end
end

switch ftype
    case {'cfs', 'mros','shhs'}
        % Read XML file
        s = xml2struct(p_file);
        events = s.CMPStudyConfig.ScoredEvents.ScoredEvent;
        ah = struct;
        ah_seq = zeros(1,L);
        k = 1;
        % Iterate over all events and save apneas/hypopneas
        apnea_str = {'Hypopnea','Apnea'};
        for i = 1:length(events)
            if contains(events{i}.Name.Text,apnea_str)
                ah.start(k) = str2num(events{i}.Start.Text);
                ah.duration(k) = str2num(events{i}.Duration.Text);
                ah.stop(k) = ah.start(k) + ah.duration(k);
                if ah.start(k) < 1
                    ah.start(k) = 1;
                end
                ah_seq(ceil(ah.start(k)):ceil(ah.stop(k))) = 1;
                k = k + 1;
            end
        end
        ah_seq = ah_seq(1:L);
    case 'wsc2'
        % Read csv annotation file
        T = readtable(p_file,'FileType','text','Delimiter',';');
        T = T(:,end-1:end);
        T.Properties.VariableNames = {'Time','Event'};
        T(cellfun(@isempty, T.Time),:) = [];
        % Get event time in seconds
        time_all = time2ind(T);
        time_ar = time_all(contains(T.Event,'Hypopnea'));
        ah = struct;
        ah_seq = zeros(1,L);
        % Iterate and save all arousals
        for i = 1:length(time_ar)
            ah.start(i) = time_ar(i);
            ah.duration(i) = 3;
            ah.stop(i) = ah.start(i) + ah.duration(i);
            ah_seq(ceil(ah.start(i)):ceil(ah.stop(i))) = 1;
        end
        ah_seq = ah_seq(1:L);
    case 'ssc'
        % Read .EVTS file
%         [SCO,~] = CLASS_codec.parseSSCevtsFile(p_file);
%         apnea_str = {'Hypopnea','Apnea'};
%         ar_idx = contains(SCO.category,apnea_str);
%         ar_event = SCO.startStopSamples(ar_idx,:)/SCO.samplerate;
        ah = struct;
        ah_seq = zeros(1,L);
        % Iterate over all arousals and save them
%         for i = 1:size(ar_event,1)
%             ah.start(i) = ar_event(i,1);
%             ah.duration(i) = ar_event(i,2)-ar_event(i,1);
%             ah.stop(i) = ah.start(i) + ah.duration(i);
%             ah_seq(ceil(ah.start(i)):ceil(ah.stop(i))) = 1;
%         end
%         ah_seq = ah_seq(1:L);
    case 'wsc'
        % Return empty labels (none exists)
        ah = struct;
        ah_seq = zeros(1,L);
    case 'stages'
        ah = struct;
        ah_seq = zeros(1,L);
end

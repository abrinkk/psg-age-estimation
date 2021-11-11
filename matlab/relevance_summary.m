%% Interpretaion stats
clear all; close all;

%% Set paths
p_base = 'H:\nAge';
p_data = 'H:\nAge\all';
p_save = 'C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data';
p_mros_anno = 'G:\mros\polysomnography\annotations-events-profusion\visit1\';
p_shhs_anno = 'H:\shhs\polysomnography\annotations-events-profusion\shhs1\';
p_ssc_anno = 'G:\ssc\polysomnography\labels\';
p_cfs_anno = 'G:\cfs\polysomnography\annotations-events-profusion\';
% p_base = '/scratch/users/abk26/nAge';
% p_data = '/scratch/users/abk26/nAge/all';
% p_save = '/home/users/abk26/SleepAge/Scripts/data';
% p_mros_anno = '/oak/stanford/groups/mignot/mros/polysomnography/annotations-events-profusion/visit1/';
% p_shhs_anno = '/home/users/abk26/SleepAge/Scripts/data/shhs/polysomnography/shhs1/';
% p_ssc_anno = '/oak/stanford/groups/mignot/psg/SSC/APOE_deid/';
% p_cfs_anno = '/oak/stanford/groups/mignot/cfs/polysomnography/annotations-events-profusion/';

m_names = {'eeg5','eegeogemg5','ecg5','resp5'};

%% Iterate interpretaion data and do summary

% Model
k = 2;

% Sampling frequency
fs = 128;

% Relevance method
rel_method = 'int_grad';

% Time-lock window
tl_window = 70*fs;

% Relevance data path
p_rel = filepath(p_base, ['model_' m_names{k}], 'interpretation', rel_method);
f_rel = dir(filepath(p_rel, '*.hdf5'));
f_rel_name = {f_rel.name};
n = size(f_rel, 1);

% Summary variables
% Average relevance in sleep stages (W, N1, N2, N3, REM),
% arousals with and without apnea,
% apnea (< 20 sec, 20 - 40, > 40 sec)
rel_ssc_avg = zeros(n,5); % (1)
rel_ar1_avg = zeros(n,2); % (1)
rel_ar2_avg = zeros(n,2); % (1)
rel_ah1_avg = zeros(n,2); % (1)
rel_ah2_avg = zeros(n,2); % (1)
rel_ah3_avg = zeros(n,2); % (1)

% Average relevance at sleep stage transitions, short arousals, long
% arousals, apnea, oxygen desautration
ssc_trans_leg = {'NREM-W','REM-W','N2-N1','N3-N2','N2-N3','N1-N2','NREM-REM','REM-NREM'};
ssc_trans = {{1,-3:-1},{1,0},{-1,-2},{-2,-3},{-3,-2},{-2,-1},{0,-3:-1},{-3:-1,0}};
rel_tl_ssc_trans = cell(n,length(ssc_trans)); % (2)
rel_tl_ar1 = cell(n,2); % (2)
rel_tl_ar2 = cell(n,2); % (2)
rel_tl_ah1 = cell(n,2); % (2)
rel_tl_ah2 = cell(n,2); % (2)
rel_tl_ah3 = cell(n,2); % (2)
evt_tl_ah1_avg = cell(n,2);
evt_tl_ah2_avg = cell(n,2);
evt_tl_ah3_avg = cell(n,2);
% EEG spectrogram scaled by relevance
rel_t1_ssc_spec = cell(n,5,2);
rel_t1_ssc_trans_spec = cell(n,length(ssc_trans),2);
rel_tl_ar1_spec = cell(n,2);
rel_tl_ar2_spec = cell(n,2);

for i = 1:n
    fprintf('Processing %.0f / %.0f\n',i,n);
    try
        % Declare paths
        record = f_rel_name{i};
        f_data = filepath(p_data, record);
        
        % Avoid if file does exist
        if ~exist(f_data,'file')
            continue
        end
        
        % Find cohort code
        if contains(record, 'cfs')
            cohort_code = 1;
        elseif length(record) == 19 || length(record) == 12
            cohort_code = 3;
        elseif contains(record, 'mros')
            cohort_code = 4;
        elseif contains(record, 'ssc')
            cohort_code = 5;
        elseif contains(record, 'shhs')
            cohort_code = 6;
        else
            cohort_code = 2;
        end
        
        % Restrict analyses to CFS, MrOS, and SHHS
        if ismember(cohort_code, [2, 3, 5])
            continue
        end
        
        % Read model interpretation
        interp = h5read(filepath(p_rel, f_rel_name{i}),'/Interpretation');
        interp = sum(interp,2);
        interp_avg = mean(reshape(interp,fs,[]),1);
        
        % Read sleep staging and PSG
        ssc = h5read(f_data,'/SSC');
        psg = h5read(f_data,'/PSG');
        if length(ssc) > (size(psg,1)/(fs*30))
            ssc = ssc(1:size(psg,1)/(fs*30));
        end
        if length(ssc) > (size(interp,1)/(fs*30))
            ssc = ssc(1:size(interp,1)/(fs*30));
        end
        ssc_up = repelem(ssc',30*fs);
        
        % Load arousal and apneas
        record_no_ext = record(1:end-5);
        if contains(record_no_ext,'cfs')
            ftype = 'cfs';
            evt_path = [p_cfs_anno record_no_ext '-profusion.xml'];
        elseif contains(record_no_ext,'SSC')
            ftype = 'ssc';
            evt_path = [p_ssc_anno record_no_ext '.evts'];
        elseif contains(record_no_ext,'shhs')
            ftype = 'shhs';
            evt_path = [p_shhs_anno record_no_ext '-profusion.xml'];
        elseif contains(record_no_ext,'EDFAndScore')
            ftype = 'stages';
            evt_path = '';
        elseif contains(record_no_ext, 'mros')
            ftype = 'mros';
            evt_path = [p_mros_anno record_no_ext '-profusion.xml'];
        else
            ftype = 'wsc';
            evt_path = '';
        end
        L = round(length(interp)/fs);
        [ar,ar_seq] = LoadAR(evt_path,L,ftype);
        [ah,~] = LoadAH(evt_path,L,ftype);
        [od,od_seq] = LoadOD(evt_path,L,ftype);
        od_seq_up = repelem(od_seq,fs);
        
        % Continue if no events
        if isempty(ah) || isempty(ar)
            continue
        end
        
        % Preprocess apneas
        idx_ah_ar_iso = any(ah.start - ar.stop' < 0 & ah.start - ar.stop' > -60,1);
        idx_ar_ah_iso = any(ar.start - ah.stop' < 0 & ar.start - ah.stop' > -60,1);
        idx_ah_iso = [true ah.start(2:end) - ah.stop(1:end-1) > 60] & (~idx_ah_ar_iso);
        idx_ar_iso = [true ar.start(2:end) - ar.stop(1:end-1) > 60] & (~idx_ar_ah_iso);
        ah_iso = struct;
        ah_iso.start = ah.start(idx_ah_iso);
        ah_iso.duration = ah.duration(idx_ah_iso);
        ah_iso.stop = ah.stop(idx_ah_iso);
        ah_iso.N = length(ah_iso.start);
        ah1 = struct;
        ah1.start = ah_iso.start(ah_iso.duration <= 20);
        ah1.duration = ah_iso.duration(ah_iso.duration <= 20);
        ah1.stop = ah_iso.stop(ah_iso.duration <= 20);
        ah1.N = length(ah1.start);
        ah1_seq = ar2sequence(ah1, 1, L);
        ah2 = struct;
        ah2.start = ah_iso.start(ah_iso.duration > 20 & ah_iso.duration <= 40);
        ah2.duration = ah_iso.duration(ah_iso.duration > 20 & ah_iso.duration <= 40);
        ah2.stop = ah_iso.stop(ah_iso.duration > 20 & ah_iso.duration <= 40);
        ah2.N = length(ah2.start);
        ah2_seq = ar2sequence(ah2, 1, L);
        ah3 = struct;
        ah3.start = ah_iso.start(ah_iso.duration > 40);
        ah3.duration = ah_iso.duration(ah_iso.duration > 40);
        ah3.stop = ah_iso.stop(ah_iso.duration > 40);
        ah3.N = length(ah3.start);
        ah3_seq = ar2sequence(ah3, 1, L);
        
        % Preprocess arousals
        idx_ar_ah = any(ar.start - ah.stop' < 10 &  ar.start - ah.stop' > -5,1);
        ar1 = struct;
        ar1.start = ar.start(~idx_ar_ah & idx_ar_iso);
        ar1.duration = ar.duration(~idx_ar_ah & idx_ar_iso);
        ar1.stop = ar.stop(~idx_ar_ah & idx_ar_iso);
        ar1.N = length(ar1.start);
        ar1_seq = ar2sequence(ar1, 1, L);
        ar2.start = ar.start(idx_ar_ah & idx_ar_iso);
        ar2.duration = ar.duration(idx_ar_ah & idx_ar_iso);
        ar2.stop = ar.stop(idx_ar_ah & idx_ar_iso);
        ar2.N = length(ar2.start);
        ar2_seq = ar2sequence(ar2, 1, L);
        
        % Monitor arousal and desaturation distribution for ah1, ah2, and ah3
        idx_ah1_offset_sec = round(ah1.stop(ah1.stop <= L-tl_window/fs & ah1.stop >= tl_window/fs));
        idx_ah2_offset_sec = round(ah2.stop(ah2.stop <= L-tl_window/fs & ah2.stop >= tl_window/fs));
        idx_ah3_offset_sec = round(ah3.stop(ah3.stop <= L-tl_window/fs & ah3.stop >= tl_window/fs));
        evt_tl_ah1_avg(i,1) = {mean(cell2mat(arrayfun(@(x) ar_seq((x-tl_window/fs+1):(x+tl_window/fs))',idx_ah1_offset_sec,'Un',0)),2)};
        evt_tl_ah1_avg(i,2) = {mean(cell2mat(arrayfun(@(x) od_seq((x-tl_window/fs+1):(x+tl_window/fs))',idx_ah1_offset_sec,'Un',0)),2)};
        evt_tl_ah2_avg(i,1) = {mean(cell2mat(arrayfun(@(x) ar_seq((x-tl_window/fs+1):(x+tl_window/fs))',idx_ah2_offset_sec,'Un',0)),2)};
        evt_tl_ah2_avg(i,2) = {mean(cell2mat(arrayfun(@(x) od_seq((x-tl_window/fs+1):(x+tl_window/fs))',idx_ah2_offset_sec,'Un',0)),2)};
        evt_tl_ah3_avg(i,1) = {mean(cell2mat(arrayfun(@(x) ar_seq((x-tl_window/fs+1):(x+tl_window/fs))',idx_ah3_offset_sec,'Un',0)),2)};
        evt_tl_ah3_avg(i,2) = {mean(cell2mat(arrayfun(@(x) od_seq((x-tl_window/fs+1):(x+tl_window/fs))',idx_ah3_offset_sec,'Un',0)),2)};
        
        % 1) Average in sleep stages, arousals, and apneas/hyponeas
        rel_ssc_avg(i,:) = arrayfun(@(x) mean(interp_avg(repelem(ssc, 30) == x)),1:-1:-3);
        rel_ar1_avg(i,:) = arrayfun(@(x) mean(interp_avg(ar1_seq == x)),[0, 1]);
        rel_ar2_avg(i,:) = arrayfun(@(x) mean(interp_avg(ar2_seq == x)),[0, 1]);
        rel_ah1_avg(i,:) = arrayfun(@(x) mean(interp_avg(ah1_seq == x)),[0, 1]);
        rel_ah2_avg(i,:) = arrayfun(@(x) mean(interp_avg(ah2_seq == x)),[0, 1]);
        rel_ah3_avg(i,:) = arrayfun(@(x) mean(interp_avg(ah3_seq == x)),[0, 1]);
        
        % 2) Timelock to sleep stage transitions
        idx_ssc_trans = cellfun(@(x) 1+find(ismember(ssc_up(2:end),x{1}) & ismember(ssc_up(1:end-1),x{2})),ssc_trans,'Un',0);
        idx_ssc_trans = cellfun(@(x) x(rem((x-1),5*60*fs) ~= 0 & x <= L*fs-tl_window & x >= tl_window),idx_ssc_trans,'Un',0);
        for j = 1:length(idx_ssc_trans)
            idx_ssc_trans_j = idx_ssc_trans{j}(rem((idx_ssc_trans{j}-1),5*60*fs) ~= 0);
            rel_tl_ssc_trans(i,j) = {mean(cell2mat(arrayfun(@(x) interp((x-tl_window+1):(x+tl_window)),idx_ssc_trans_j,'Un',0)),2)};
        end
        
        % 2) Timelock to arousal and apnea/hypopnea onset and offset
        idx_ar1_onset = round(ar1.start * fs);
        idx_ar1_offset = round(ar1.stop * fs);
        idx_ar1_onset_2 = idx_ar1_onset(rem(idx_ar1_onset, 5*60*fs) >= 30*fs & rem(idx_ar1_offset, 5*60*fs) <= (5*60-30)*fs & idx_ar1_offset <= L*fs-tl_window & idx_ar1_onset >= tl_window & [true diff(idx_ar1_onset) > 60*fs]);
        idx_ar1_offset_2 = idx_ar1_offset(rem(idx_ar1_onset, 5*60*fs) >= 30*fs & rem(idx_ar1_offset, 5*60*fs) <= (5*60-30)*fs & idx_ar1_offset <= L*fs-tl_window & idx_ar1_onset >= tl_window & [true diff(idx_ar1_offset) > 60*fs]);
        rel_tl_ar1(i,1) = {mean(cell2mat(arrayfun(@(x) interp((x-tl_window+1):(x+tl_window)),idx_ar1_onset_2,'Un',0)),2)};
        rel_tl_ar1(i,2) = {mean(cell2mat(arrayfun(@(x) interp((x-tl_window+1):(x+tl_window)),idx_ar1_offset_2,'Un',0)),2)};
        idx_ar2_onset = round(ar2.start * fs);
        idx_ar2_offset = round(ar2.stop * fs);
        idx_ar2_onset_2 = idx_ar2_onset(rem(idx_ar2_onset, 5*60*fs) >= 30*fs & rem(idx_ar2_offset, 5*60*fs) <= (5*60-30)*fs & idx_ar2_offset <= L*fs-tl_window & idx_ar2_onset >= tl_window & [true diff(idx_ar2_onset) > 60*fs]);
        idx_ar2_offset_2 = idx_ar2_offset(rem(idx_ar2_onset, 5*60*fs) >= 30*fs & rem(idx_ar2_offset, 5*60*fs) <= (5*60-30)*fs & idx_ar2_offset <= L*fs-tl_window & idx_ar2_onset >= tl_window & [true diff(idx_ar2_offset) > 60*fs]);
        rel_tl_ar2(i,1) = {mean(cell2mat(arrayfun(@(x) interp((x-tl_window+1):(x+tl_window)),idx_ar2_onset_2,'Un',0)),2)};
        rel_tl_ar2(i,2) = {mean(cell2mat(arrayfun(@(x) interp((x-tl_window+1):(x+tl_window)),idx_ar2_offset_2,'Un',0)),2)};
        
        idx_ah1_onset = round(ah1.start * fs);
        idx_ah1_offset = round(ah1.stop * fs);
        idx_ah1_onset_2 = idx_ah1_onset(rem(idx_ah1_onset, 5*60*fs) >= 30*fs & rem(idx_ah1_offset, 5*60*fs) <= (5*60-30)*fs & idx_ah1_offset <= L*fs-tl_window & idx_ah1_onset >= tl_window);
        idx_ah1_offset_2 = idx_ah1_offset(rem(idx_ah1_onset, 5*60*fs) >= 30*fs & rem(idx_ah1_offset, 5*60*fs) <= (5*60-30)*fs & idx_ah1_offset <= L*fs-tl_window & idx_ah1_onset >= tl_window);
        rel_tl_ah1(i,1) = {mean(cell2mat(arrayfun(@(x) interp((x-tl_window+1):(x+tl_window)),idx_ah1_onset_2,'Un',0)),2)};
        rel_tl_ah1(i,2) = {mean(cell2mat(arrayfun(@(x) interp((x-tl_window+1):(x+tl_window)),idx_ah1_offset_2,'Un',0)),2)};
        idx_ah2_onset = round(ah2.start * fs);
        idx_ah2_offset = round(ah2.stop * fs);
        idx_ah2_onset_2 = idx_ah2_onset(rem(idx_ah2_onset, 5*60*fs) >= 30*fs & rem(idx_ah2_offset, 5*60*fs) <= (5*60-30)*fs & idx_ah2_offset <= L*fs-tl_window & idx_ah2_onset >= tl_window);
        idx_ah2_offset_2 = idx_ah2_offset(rem(idx_ah2_onset, 5*60*fs) >= 30*fs & rem(idx_ah2_offset, 5*60*fs) <= (5*60-30)*fs & idx_ah2_offset <= L*fs-tl_window & idx_ah2_onset >= tl_window);
        rel_tl_ah2(i,1) = {mean(cell2mat(arrayfun(@(x) interp((x-tl_window+1):(x+tl_window)),idx_ah2_onset_2,'Un',0)),2)};
        rel_tl_ah2(i,2) = {mean(cell2mat(arrayfun(@(x) interp((x-tl_window+1):(x+tl_window)),idx_ah2_offset_2,'Un',0)),2)};
        idx_ah3_onset = round(ah3.start * fs);
        idx_ah3_offset = round(ah3.stop * fs);
        idx_ah3_onset_2 = idx_ah3_onset(rem(idx_ah3_onset, 5*60*fs) >= 30*fs & rem(idx_ah3_offset, 5*60*fs) <= (5*60-30)*fs & idx_ah3_offset <= L*fs-tl_window & idx_ah3_onset >= tl_window);
        idx_ah3_offset_2 = idx_ah3_offset(rem(idx_ah3_onset, 5*60*fs) >= 30*fs & rem(idx_ah3_offset, 5*60*fs) <= (5*60-30)*fs & idx_ah3_offset <= L*fs-tl_window & idx_ah3_onset >= tl_window);
        rel_tl_ah3(i,1) = {mean(cell2mat(arrayfun(@(x) interp((x-tl_window+1):(x+tl_window)),idx_ah3_onset_2,'Un',0)),2)};
        rel_tl_ah3(i,2) = {mean(cell2mat(arrayfun(@(x) interp((x-tl_window+1):(x+tl_window)),idx_ah3_offset_2,'Un',0)),2)};
        
        % 3) Timelock to arousal and apnea/hypopnea with spectrogram
        if ismember(k, [1 2])
            eeg1 = psg(:,1);
            eeg2 = psg(:,2);
            [~,f1,t1,ps1] = spectrogram(eeg1,3*fs,2*fs,0:0.1:fs/2,fs,'yaxis');
            [~,f2,t2,ps2] = spectrogram(eeg2,3*fs,2*fs,0:0.1:fs/2,fs,'yaxis');
            ps1 = (ps1/mean(eeg1.^2) + ps2/mean(eeg2.^2))/(2);
            ps1_crop = ([zeros(length(f1),1) ps1(:,1:(length(interp_avg) - 2)), zeros(length(f1),1)]);
            rel_ps1 = ps1_crop .* repmat(interp_avg,length(f1),1);
            idx_rel_pos = interp_avg >= 0;
            rel_t1_ssc_spec(i,:,1) = arrayfun(@(x) mean(ps1_crop(:,repelem(ssc,30) == x),2),1:-1:-3,'Un',0);
            rel_t1_ssc_spec(i,:,2) = arrayfun(@(x) mean(rel_ps1(:,repelem(ssc,30) == x),2),1:-1:-3,'Un',0);
            
            rel_tl_ar1_spec(i,1) = {mean(reshape(cell2mat(arrayfun(@(x) ps1_crop(:,(x-tl_window/fs+1):(x+tl_window/fs)),round(idx_ar1_onset_2/fs),'Un',0)),length(f1),2*tl_window/fs,[]),3)};
            rel_tl_ar1_spec(i,2) = {mean(reshape(cell2mat(arrayfun(@(x) rel_ps1(:,(x-tl_window/fs+1):(x+tl_window/fs)),round(idx_ar1_onset_2/fs),'Un',0)),length(f1),2*tl_window/fs,[]),3)};
            rel_tl_ar2_spec(i,1) = {mean(reshape(cell2mat(arrayfun(@(x) ps1_crop(:,(x-tl_window/fs+1):(x+tl_window/fs)),round(idx_ar2_onset_2/fs),'Un',0)),length(f1),2*tl_window/fs,[]),3)};
            rel_tl_ar2_spec(i,2) = {mean(reshape(cell2mat(arrayfun(@(x) rel_ps1(:,(x-tl_window/fs+1):(x+tl_window/fs)),round(idx_ar2_onset_2/fs),'Un',0)),length(f1),2*tl_window/fs,[]),3)};
        end
    catch me
        disp(me.message);
    end
end

g_win = gausswin(fs * 10, (fs * 10)/(30 * 2));
rel_tl_ssc_trans_sum = arrayfun(@(x) summarize_rel_stat(rel_tl_ssc_trans(:,x), g_win, 2), 1:size(rel_tl_ssc_trans,2), 'Un', 0);
rel_tl_ar1_sum = arrayfun(@(x) summarize_rel_stat(rel_tl_ar1(:,x), g_win, 2), 1:size(rel_tl_ar1,2), 'Un', 0);
rel_tl_ar2_sum = arrayfun(@(x) summarize_rel_stat(rel_tl_ar2(:,x), g_win, 2), 1:size(rel_tl_ar2,2), 'Un', 0);
rel_tl_ah1_sum = arrayfun(@(x) summarize_rel_stat(rel_tl_ah1(:,x), g_win, 2), 1:size(rel_tl_ah1,2), 'Un', 0);
rel_tl_ah2_sum = arrayfun(@(x) summarize_rel_stat(rel_tl_ah2(:,x), g_win, 2), 1:size(rel_tl_ah2,2), 'Un', 0);
rel_tl_ah3_sum = arrayfun(@(x) summarize_rel_stat(rel_tl_ah3(:,x), g_win, 2), 1:size(rel_tl_ah3,2), 'Un', 0);
evt_tl_ah1_avg_sum = arrayfun(@(x) summarize_rel_stat(evt_tl_ah1_avg(:,x), 1, 2), 1:size(evt_tl_ah1_avg,2), 'Un', 0);
evt_tl_ah2_avg_sum = arrayfun(@(x) summarize_rel_stat(evt_tl_ah2_avg(:,x), 1, 2), 1:size(evt_tl_ah2_avg,2), 'Un', 0);
evt_tl_ah3_avg_sum = arrayfun(@(x) summarize_rel_stat(evt_tl_ah3_avg(:,x), 1, 2), 1:size(evt_tl_ah3_avg,2), 'Un', 0);
ssc_spec_sum = arrayfun(@(x) summarize_rel_stat(rel_t1_ssc_spec(:,x,1), 1, 2), 1:size(rel_t1_ssc_spec,2), 'Un', 0);
rel_t1_ssc_spec_sum = arrayfun(@(x) summarize_rel_stat(rel_t1_ssc_spec(:,x,2), 1, 2), 1:size(rel_t1_ssc_spec,2), 'Un', 0);
% TODO: MAKE THIS CODE BELOW WORK
rel_tl_ar1_spec_sum = arrayfun(@(x) summarize_rel_stat(rel_tl_ar1_spec(:,x), 1, 3), 1:size(rel_tl_ar1_spec,2), 'Un', 0);
rel_tl_ar2_spec_sum = arrayfun(@(x) summarize_rel_stat(rel_tl_ar2_spec(:,x), 1, 3), 1:size(rel_tl_ar2_spec,2), 'Un', 0);


% save(filepath(p_save, sprintf('relevance_stats_model %.0f.mat', k)), ...
%     'f_rel_name', 'rel_tl_ssc_trans_sum', 'rel_tl_ar1_sum', 'rel_tl_ar2_sum', ...
%     'rel_tl_ah1_sum', 'rel_tl_ah2_sum', 'rel_tl_ah3_sum','ssc_trans_leg',...
%     'evt_tl_ah1_avg_sum','evt_tl_ah2_avg_sum','evt_tl_ah3_avg_sum','ssc_spec_sum',...
%     'rel_t1_ssc_spec_sum','rel_tl_ar1_spec_sum',...
%     'rel_tl_ar2_spec_sum');
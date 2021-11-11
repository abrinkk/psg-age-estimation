clear all; close all;
addpath(genpath('export_fig'));
addpath(genpath('edfread'));

%% Load Data

% Select channels
k = 1;
m_names = {'eeg5','eegeogemg5','ecg5','resp5'};
m_run = m_names{k};
with_rel = true;
rel_method = 'grad_shap';

% Paths
path_psg = 'H:\nAge\all\';
path_int = ['H:\nAge\model_' m_run '\interpretation\' rel_method '\'];

switch m_run
    case 'eeg5'
        names_chan = {'C_3-A_2','C_4-A_1'};
        idx_chan = 1:2;
        idx_epoch = 69;
        offset = 0;
        dur = 90;
        % Select record
        record = 'cfs-visit5-800105.hdf5';
    case 'eegeogemg5'
        names_chan = {'C_3-A_2','C_4-A_1','EOG_L','EOG_R','Chin EMG'};
        idx_chan = [1:4 6];
        idx_epoch = 69;
        offset = 0;
        dur = 90;
        % Select record
        record = 'cfs-visit5-800105.hdf5';
    case 'ecg5'
        names_chan = {'ECG'};
        idx_chan = 5;
        idx_epoch = 63;
        offset = 235;
        dur = 30;
        % Select record
        record = 'cfs-visit5-800914.hdf5';
    case 'resp5'
        names_chan = {'Airflow','Nasal Pressure','Abdomen','Thorax','SaO2'};
        idx_chan = 8:12;
        idx_epoch = 55;
        offset = 0;
        dur = 300;
        % Select record
        record = 'cfs-visit5-800820.hdf5';
    case '5'
        names_chan = {'C_3-A_2','C_4-A_1','EOG_L','EOG_R','ECG','Chin EMG','Leg','Airflow','Nasal Pressure','Abdomen','Thorax','SaO2'};
        idx_chan = 1:12;
        idx_epoch = 71;
        offset = 150;
        dur = 60;
        % Select record
        record = 'cfs-visit5-800105.hdf5';
end

% Load h5 file
rel = h5read([path_int record],'/Interpretation');
psg = h5read([path_psg record],'/PSG');
psg = psg(1:size(rel, 1), idx_chan);

%% Rescale signals
idx_sao2 = strcmp(names_chan, 'SaO2');
psg(:,idx_sao2) = psg(:,idx_sao2)*10 - 7.5;

%% Heart rate
if any(strcmp(names_chan, 'ECG'))
    record_edf_folder = 'G:\cfs\polysomnography\edfs\';
    edf_filename = [record_edf_folder, record(1:end-5), '.edf'];
    [hdr, edf_data] = edfread(edf_filename);
    HR_idx = strcmp(hdr.label, 'PULSE');
    HR = edf_data(HR_idx,:);
    clear edf_data;
else
    HR = -1;
end

%% Smooth and max pool relevance attributions
fs = 128;
switch m_run
    case 'eeg5'
        g_std = 30;
    case 'eegeogemg5'
        g_std = 30;
    case 'ecg5'
        g_std = 30;
    case 'resp5'
        g_std = 120;
    case '5'
        g_std = 30;
end
g_win = gausswin(fs * 10, (fs * 10)/(g_std * 2));
rel_g = conv2(rel, g_win, 'same') / sum(g_win);
mp_size = [fs / 8, 1];
rel_mp = sepblockfun(rel_g, mp_size, 'mean');
fs_mp = 8;

%% Data visualization
dur_s = dur * fs;
dur_r = dur * fs_mp;
epoch_size = 5*60*fs;
t = 0:1/fs:dur_s/fs-1/fs;
dy_inc = 10;
dy = dy_inc*(0:length(idx_chan)-1);
pos = [0.1300    0.7093    0.7750    0.2157];
pos = [pos(1)    pos(2) + pos(4)*(1 - (2 + 1)/(size(rel_mp,2) + with_rel))   pos(3)*0.8    pos(4)*(2 + 1)/(size(rel_mp,2) + with_rel)];

h = figure;
h.Position(3:4) = [1000 80 + 40*(size(rel_mp,2) + with_rel)];
centerfig(h);
hold all
% Input relevance
if with_rel
    C = load("C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\matlab\RdBu_r.txt", '-ascii');
    C = C(:,1:3);
    idx_rel_mp = (idx_epoch - 1)*epoch_size/fs*fs_mp + offset*fs_mp;
    x_im = 0:1/fs_mp:dur_r/fs_mp - 1/fs_mp;
%     ax1 = subplot((size(rel_mp,2) + with_rel),1,1);
    ax1 = subplot('position',pos);
    hold all
    plot([0 dur_r/fs_mp - 1/fs_mp], [0 0] ,'--','Color',[0.6 0.6 0.6])
    p_rel_bar = plot(x_im, sum(rel_mp((1+idx_rel_mp):(idx_rel_mp + dur_r),:),2),'k');
    ax1.YAxis.Exponent = 0;
%     xtickformat('%.5f');
    grid minor
    set(gca,'XTickLabel',{});
    box on
    ylim([-max(abs(get(gca,'YLim'))) max(abs(get(gca,'YLim')))])
%     cb_invis = colorbar;
%     set(cb_invis, 'YTickLabel', cellstr(num2str(reshape(get(cb_invis, 'YTick'),[],1),'%0.3g')) )
%     set(get(cb_invis,'Label'), 'String', {'Relevance','Score'});
%     ax2 = subplot((size(rel_mp,2) + with_rel),1,2:(size(rel_mp,2) + with_rel));
    ax2 = subplot('position',[pos(1) pos(2)-pos(4)*size(rel_mp,2)-0.01 pos(3) pos(4)*size(rel_mp,2)]);
    hold all
    colormap(C);
    if size(rel_mp,2) == 1
        y_im = [ - dy(end) - dy_inc/2, -dy(1) + dy_inc/2];
    else
        y_im = 0:-1:-(size(rel_mp,2)-1)*dy_inc;
    end
    imagesc(x_im, y_im, rel_mp((1+idx_rel_mp):(idx_rel_mp + dur_r),:)');
    % imagesc(t, ...
    %         (0:-1:-(size(rel_mp,2)-1))*dy_inc, ...
    %         rel_g(1 + idx_psg:(idx_psg + dur_s), :)');
    cb = colorbar;
    pos2 = ax2.Position;
    cb.Position = [pos2(1) + pos2(3) + 0.01 pos2(2) cb.Position(3) cb.Position(4)];
%     ax1.Position = [ax2.Position(1) ax2.Position(2) + ax2.Position(4) + 0.01 ax2.Position(3) ax1.Position(4)];
%     cb.Position(4) = ax1.Position(2) + ax1.Position(4) - ax2.Position(2);
%     ax1.Position(3) = cb.Position(1) - ax2.Position(1);
    rel_std = std(reshape(rel_mp((1+idx_rel_mp):(idx_rel_mp + dur_r),:)',1,[]));
    caxis([-10*rel_std 10*rel_std])
    set(cb, 'YTickLabel', cellstr(num2str(reshape(get(cb, 'YTick'),[],1),'%0.3g')) )
    set(get(cb,'Label'), 'String', {'Relevance','Score'});
end
% HR data
if length(HR) ~= 1
    idx_chan_ecg = strcmp(names_chan, 'ECG');
    idx_hr = (idx_epoch - 1)*epoch_size/fs + offset;
    for i = 1:2:dur
        text(i, 0.3*dy_inc - dy(idx_chan_ecg), num2str(HR(i + idx_hr),'%.0f'), 'HorizontalAlignment', 'center');
    end
end
% PSG data
for i = 1:length(idx_chan)
    idx_psg = (idx_epoch - 1)*epoch_size + offset*fs;
    plot(t, psg(1 + idx_psg:(idx_psg + dur_s), i) - dy(i), 'k', 'LineWidth', 0.5);
end
grid minor;
set(gca,'YTick',fliplr(-dy));
set(gca,'YTickLabel',fliplr(names_chan));
axis([0, dur_s/fs, - dy(end) - dy_inc/2, -dy(1) + dy_inc/2]);
% Plot box around
plot([0, dur_s/fs], ones(2,1)*(-dy(1) + dy_inc/2),'k', 'LineWidth', 0.5);
plot([0, dur_s/fs], ones(2,1)*(- dy(end) - dy_inc/2),'k', 'LineWidth', 0.5);
plot(ones(2,1)*(dur_s/fs), [- dy(end) - dy_inc/2, -dy(1) + dy_inc/2],'k', 'LineWidth', 0.5);
plot(ones(2,1)*(0), [ - dy(end) - dy_inc/2, -dy(1) + dy_inc/2],'k', 'LineWidth', 0.5);

xlabel('Time [s]');
ylabel('');
set(gcf,'Color',[1 1 1]);
set( findall(h, '-property', 'fontsize'), 'fontsize', 10);
export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\relevance_' m_run '_rel_bar_' num2str(with_rel) '_rel_method_ ' rel_method], '-pdf', '-transparent');

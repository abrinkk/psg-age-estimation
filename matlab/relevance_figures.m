%% Arousal and desaturation after apnea
clear all; close all
addpath(genpath('export_fig'));

% TODO: Load all summary and do two figures
%       1) Model 2 Arousal onset + Model 4 apnea (1/2/3) onset
%       2) Model 1, 2, 3 Arousal onset + Model 4 apnea (1+2+3) onset 
 
%% Load data
rel = cell(4,1);
rel_method = 'grad_shap';
fs = 128;
tl_window = 70*fs;
names_plot = {'\bf{a}\rm Central EEG','\bf{b}\rm EEG+EOG+EMG','\bf{c}\rm ECG','\bf{d}\rm Respiratory'};
for k = 1:4
    rel{k} = load(['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data\relevance_stats_model ' num2str(k) ' ' rel_method '.mat']);
end
load('C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data\X_test.mat');

%% Relevance time-lock plot

xl = [-20 20];
t = (-tl_window+1):tl_window;
idx = t > xl(1)*fs & t <= xl(2)*fs;
h = figure;
h.Position(3:4) = [1000 600];
centerfig(h);
for k = 1:4
    if ismember(k, 1:3)
        rel_tl_ar1_sum = rel{k}.rel_tl_ar1_sum;
        leg = {'Arousal Onset'};
    elseif k == 4
        rel_tl_ar1_sum = rel{k}.rel_tl_ah1_sum;
        leg = {'Apnea/hypopnea Onset'};
    end
    subplot(4,1,k)
    hold all
    p1 = plot(((-tl_window+1):tl_window)/fs, rel_tl_ar1_sum{1}{1} * 10^4,'Color',[0.6350, 0.0780, 0.1840],'LineWidth',1.5);
    p2 = plot(((-tl_window+1):tl_window)/fs, (rel_tl_ar1_sum{1}{1} + rel_tl_ar1_sum{1}{2} / sqrt(sum(rel_tl_ar1_sum{1}{3}))) * 10^4, '--','Color',[0.8500, 0.3250, 0.0980]);
    plot(((-tl_window+1):tl_window)/fs, (rel_tl_ar1_sum{1}{1} - rel_tl_ar1_sum{1}{2} / sqrt(sum(rel_tl_ar1_sum{1}{3}))) * 10^4, '--','Color',[0.8500, 0.3250, 0.0980]);
    % yl = get(gca,'YLim');
    yl = double(prctile(rel_tl_ar1_sum{1}{1}(idx) * 10^4, [1 99]) + [-1.5 1.5]);
    plot([0 0], yl,'--k')
    set(gca,'YLim',yl);
    grid on
    xlim(xl);
%     text(xl(1) + diff(xl)*0.05, yl(2) - diff(yl)*0.05, leg{1});
    title( leg{1});
    ylabel({[names_plot{k}],'Avg. Relevance 10^{-4}'});
    if k == 4 
        xlabel('Time [s]');
    end
end
legend([p1 p2], {'Avg. Relevance', sprintf('\x00B1 SEM')}, 'Location','southeast')
set(gcf,'Color',[1 1 1]);
set( findall(h, '-property', 'fontsize'), 'fontsize', 10);
% export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\event_tl_relevance' '_m_' rel_method], '-pdf', '-transparent');

%%
leg = {'Apnea/hypopnea Onset'};
rel_tl_ah1_sum = rel{k}.rel_tl_ah1_sum;
h = figure;
h.Position(3:4) = [600 300];
centerfig(h);
hold all
p1 = plot(((-tl_window+1):tl_window)/fs, rel_tl_ah1_sum{1}{1} * 10^4,'Color',[0.6350, 0.0780, 0.1840],'LineWidth',1.5);
p2 = plot(((-tl_window+1):tl_window)/fs, (rel_tl_ah1_sum{1}{1} + rel_tl_ah1_sum{1}{2} / sqrt(sum(rel_tl_ah1_sum{1}{3}))) * 10^4, '--','Color',[0.8500, 0.3250, 0.0980]);
plot(((-tl_window+1):tl_window)/fs, (rel_tl_ah1_sum{1}{1} - rel_tl_ah1_sum{1}{2} / sqrt(sum(rel_tl_ah1_sum{1}{3}))) * 10^4, '--','Color',[0.8500, 0.3250, 0.0980]);
% yl = get(gca,'YLim');
yl = prctile(rel_tl_ah1_sum{1}{1} * 10^4, [5 95]) + [-0.5 0.5];
plot([0 0], yl,'--k')
set(gca,'YLim',yl);
grid minor
xlim([-10 30]);
title(leg{1});
xlabel('Time [s]');
ylabel({[names_plot{k}],'Avg. Relevance 10^{-4}'});
legend([p1 p2], {'Avg. Relevance', sprintf('\x00B1 SEM')}, 'Location','southeast')
set(gcf,'Color',[1 1 1]);
set( findall(h, '-property', 'fontsize'), 'fontsize', 10);
% export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\event_tl_relevance_g_' names_plot{k} '_m_' rel_method], '-pdf', '-transparent');

%%
k = 2;
ssc_trans_leg = {'NREM-W','REM-W','N2-N1','N3-N2','N2-N3','N1-N2','NREM-REM','REM-NREM'};
rel_tl_ssc_trans_sum = rel{k}.rel_tl_ssc_trans_sum;
h = figure;
h.Position(3:4) = [800 300];
centerfig(h);
xl = [-30 30];
t = (-tl_window+1):tl_window;
idx = t > xl(1)*fs & t <= xl(2)*fs;

for i = 1:size(rel_tl_ssc_trans_sum,2)
    subplot(2,size(rel_tl_ssc_trans_sum,2)/2,i)
    hold all
    p1 = plot(((-tl_window+1):tl_window)/fs, rel_tl_ssc_trans_sum{i}{1} * 10^4,'Color',[0.6350, 0.0780, 0.1840],'LineWidth',1.5);
    p2 = plot(((-tl_window+1):tl_window)/fs, (rel_tl_ssc_trans_sum{i}{1} + rel_tl_ssc_trans_sum{i}{2} / sqrt(sum(rel_tl_ssc_trans_sum{i}{3}))) * 10^4, '--','Color',[0.8500, 0.3250, 0.0980]);
    plot(((-tl_window+1):tl_window)/fs, (rel_tl_ssc_trans_sum{i}{1} - rel_tl_ssc_trans_sum{i}{2} / sqrt(sum(rel_tl_ssc_trans_sum{i}{3}))) * 10^4, '--','Color',[0.8500, 0.3250, 0.0980]);
%     plot(((-tl_window+1):tl_window)/fs, rel_tl_ssc_trans_sum{i}{1} * 10^4);
    yl = prctile(rel_tl_ssc_trans_sum{i}{1}(idx) * 10^4, [5 95]) + [-1 1];
    plot([0 0], yl,'--k')
    set(gca,'YLim',yl);
    grid on
    xlim(xl);
    title(ssc_trans_leg{i});
    if i > 4
        xlabel('Time [s]');
    end
    if i == 1 || i == 5
        ylabel({'Avg. Relevance 10^{-4}'});
    end
end
% legend([p1 p2], {'Avg. Relevance', sprintf('\x00B1 SEM')}, 'Location','eastoutside')
set(gcf,'Color',[1 1 1]);
set( findall(h, '-property', 'fontsize'), 'fontsize', 10);
% export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\sleep_stage_transition_tl_relevance_g_' names_plot{k} '_m_' rel_method], '-pdf', '-transparent');

%% Relevance spectrogram plot
f1 = linspace(0,fs/2-1/fs,641);
symlog_trans = @(x, C) sign(x) .* log10(1 + abs(x)/(10^C));
% idx_1 = cellfun(@(x) ~isempty(x), rel_tl_ar1_spec);
% rel_tl_ar1_spec_avg = mean(reshape(cell2mat(cellfun(@(x) single(x), rel_tl_ar1_spec(idx_1(:,2) & idx_age,2)', 'Un', 0)),length(f1),2*tl_window/fs,[]),3,'omitnan');
rel_tl_ar2_spec_sum = rel{k}.rel_tl_ar2_spec_sum;
h = figure;
h.Position(3:4) = [800 300];
centerfig(h);
C_const = -6;
CData = symlog_trans(rel_tl_ar2_spec_sum{2}{1}, C_const);
imagesc((-tl_window/fs-1):tl_window/fs,f1,CData)
axis([-40 40 0 16])
set(gca,'YDir','default');
xlabel('Time [s]')
ylabel('Frequency [Hz]')
C = load("C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\matlab\RdBu_r.txt", '-ascii');
C = C(:,1:3);
if min(CData) >= 0
    C = C(256/2:256,:);
    caxis([min(CData,[],'all') max(CData,[],'all')]);
else
    caxis([-max(abs(CData),[],'all') max(abs(CData),[],'all')]);
end
colormap(C);
cb = colorbar;
set(cb, 'YTickLabel', cellstr(num2str(reshape(get(cb, 'YTick'),[],1),'%0.3g')) )
if min(CData) >= 0
    set(get(cb,'Label'), 'String', {['PSD ' '10^{' num2str(C_const) '}']});
else
    set(get(cb,'Label'), 'String', {['Relevance * PSD ' '10^{' num2str(C_const) '}']});
end
set(gcf,'Color',[1 1 1]);
set( findall(h, '-property', 'fontsize'), 'fontsize', 10);

%%
f1 = linspace(0,fs/2-1/fs,641);
% idx_1 = cellfun(@(x) ~isempty(x), rel_t1_ssc_spec);
% rel_t1_ssc_spec_avg = mean(cell2mat(cellfun(@(x) single(x), rel_t1_ssc_spec(idx_1(:,4,2) & idx_age,4,2)', 'Un', 0)),2,'omitnan');
rel_t1_ssc_spec_sum = rel{k}.rel_t1_ssc_spec_sum;
h = figure;
h.Position(3:4) = [800 300];
centerfig(h);
plot(f1,rel_t1_ssc_spec_sum{4}{1})
hold on
plot(f1([1 end]), [0 0], '--k')
symlog(gca,'y',-6);
xlim([0 45])
set(gca,'YDir','default');
xlabel('Frequency [Hz]')
if min(rel_t1_ssc_spec_sum{4}{1}) >= 0
    ylabel('PSD');
else
    ylabel('Relevance * PSD');
end
set(gcf,'Color',[1 1 1]);
set( findall(h, '-property', 'fontsize'), 'fontsize', 10);

%% Apnea duration event assocation with arousal and hypoxia


h = figure;
h.Position(3:4) = [800 800];
centerfig(h);
subplot(3,2,1)
hold all
idx_data = cellfun(@(x) ~isempty(x), evt_tl_ah1_avg(:,1)) & idx_age;
plot(-tl_window/fs+1:tl_window/fs,mean(cell2mat(evt_tl_ah1_avg(idx_data,1)'),2))
plot([0 0],[0 1],'--k');
axis([-40 40 0 1]);
grid minor
xlabel('Time from apnea offset [s]')
ylabel('Arousal [%]')
title('Apnea/Hypopnea < 20 sec')
subplot(3,2,2)
hold all
idx_data = cellfun(@(x) ~isempty(x), evt_tl_ah1_avg(:,2)) & idx_age;
plot(-tl_window/fs+1:tl_window/fs,mean(cell2mat(evt_tl_ah1_avg(idx_data,2)'),2))
plot([0 0],[0 1],'--k');
axis([-40 40 0 1]);
grid minor
xlabel('Time from apnea offset [s]')
ylabel('Hypoxia [%]')
title('Apnea/Hypopnea < 20 sec')
subplot(3,2,3)
hold all
idx_data = cellfun(@(x) ~isempty(x), evt_tl_ah2_avg(:,1)) & idx_age;
plot(-tl_window/fs+1:tl_window/fs,mean(cell2mat(evt_tl_ah2_avg(idx_data,1)'),2))
plot([0 0],[0 1],'--k');
axis([-40 40 0 1]);
grid minor
xlabel('Time from apnea offset [s]')
ylabel('Arousal [%]')
title('Apnea/Hypopnea 20 - 40 sec')
subplot(3,2,4)
hold all
idx_data = cellfun(@(x) ~isempty(x), evt_tl_ah2_avg(:,2)) & idx_age;
plot(-tl_window/fs+1:tl_window/fs,mean(cell2mat(evt_tl_ah2_avg(idx_data,2)'),2))
plot([0 0],[0 1],'--k');
axis([-40 40 0 1]);
grid minor
xlabel('Time from apnea offset [s]')
ylabel('Hypoxia [%]')
title('Apnea/Hypopnea 20 - 40 sec')
subplot(3,2,5)
hold all
idx_data = cellfun(@(x) ~isempty(x), evt_tl_ah3_avg(:,1)) & idx_age;
plot(-tl_window/fs+1:tl_window/fs,mean(cell2mat(evt_tl_ah3_avg(idx_data,1)'),2))
plot([0 0],[0 1],'--k');
axis([-40 40 0 1]);
grid minor
xlabel('Time from apnea offset [s]')
ylabel('Arousal [%]')
title('Apnea/Hypopnea >40 sec')
subplot(3,2,6)
hold all
idx_data = cellfun(@(x) ~isempty(x), evt_tl_ah3_avg(:,2)) & idx_age;
plot(-tl_window/fs+1:tl_window/fs,mean(cell2mat(evt_tl_ah3_avg(idx_data,2)'),2))
plot([0 0],[0 1],'--k');
axis([-40 40 0 1]);
grid minor
xlabel('Time from apnea offset [s]')
ylabel('Hypoxia [%]')
title('Apnea/Hypopnea >40 sec')
set(gcf,'Color',[1 1 1]);
set( findall(h, '-property', 'fontsize'), 'fontsize', 10);

%% Avg. rel in sleep stages and arousals
idx = ~all(rel_ssc_avg == 0,2);
rel_ssc_avg_all = mean(rel_ssc_avg(idx,:),1,'omitnan');
rel_ar1_avg_all = mean(rel_ar1_avg(idx,:),1,'omitnan');
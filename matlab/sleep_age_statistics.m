%% Sleep Age Statistics
clear all; close all;
addpath(genpath('export_fig'));
% addpath(genpath('sasread'));

%% Read all PSG stats
%  PSG data
X_test = readtable('H:\nAge\X_test.csv');
X_test.mode = 2*ones(size(X_test,1),1);
X_val = readtable('H:\nAge\X_val.csv');
X_val.mode = 1*ones(size(X_val,1),1);
X_train = readtable('H:\nAge\X_train.csv');
X_train.mode = 0*ones(size(X_train,1),1);
X_test = [X_test; X_val; X_train];
X_test = sortrows(X_test,find(ismember(X_test.Properties.VariableNames, 'cohort_code')));
%  Read dataset
ds_ssc_path = "H:\SSC\ssc.xlsx";
ds_ssc = readtable(ds_ssc_path,'Sheet','ssc');
ds_stages_path = 'H:\\STAGES\\PatientDemographicsAll.xlsx';
ds_stages = readtable(ds_stages_path);
ds_stages_old_path = 'H:\STAGES\PatientDemographics.xlsx';
ds_stages_old = readtable(ds_stages_old_path);
ds_stages_old = ds_stages_old(:,[2, 19]);
ds_cfs_path = 'G:\\cfs\\datasets\\cfs-visit5-dataset-0.4.0.csv';
ds_cfs = readtable(ds_cfs_path);
ds_mros_path = 'G:\\mros\\datasets\\mros-visit1-dataset-0.3.0.csv';
ds_mros = readtable(ds_mros_path);
ds_mros_dth_path = 'G:\BRINKKJAER\datasets\ap942.csv';
ds_mros_dth = readtable(ds_mros_dth_path);
ds_wsc_path = 'G:\\WSC_PLM_ data_all.xlsx';
ds_wsc = readtable(ds_wsc_path);
ds_wsc_dth_path = 'G:\wsc\WSC_all_cause_imputed.txt';
ds_wsc_dth = readtable(ds_wsc_dth_path);
ds_wsc_dth = ds_wsc_dth(:,2:end);
fid = fopen(ds_wsc_dth_path);
varNames = strsplit(fgetl(fid), '\t');
fclose(fid);
varNames = cellfun(@(x) x(2:end-1), varNames, 'Un', 0);
ds_wsc_dth.Properties.VariableNames = varNames;
ds_shhs_path = 'H:\\shhs\\datasets\\shhs1-dataset-0.15.0.csv';
ds_shhs = readtable(ds_shhs_path);
ds_shhs_cvd_path = 'H:\shhs\datasets\shhs-cvd-summary-dataset-0.15.0.csv';
ds_shhs_cvd = readtable(ds_shhs_cvd_path);
ds_sof_path = 'H:\sof\datasets\sof-visit-8-dataset-0.6.0.csv';
ds_sof = readtable(ds_sof_path);
ds_hpap_path = "H:\homepap\datasets\homepap-baseline-dataset-0.1.0.csv";
ds_hpap = readtable(ds_hpap_path);

% add black to ds_wsc_dth
ds_wsc_dth_class = cellfun(@(x) class(x), table2cell(ds_wsc_dth(1,:)), 'Un', 0);
blank_line = {};
for i = 1:length(ds_wsc_dth_class)
    if strcmp(ds_wsc_dth_class{i}, 'char')
        new_blank = '';
    elseif strcmp(ds_wsc_dth_class{i}, 'double')
        new_blank = nan;
    end
    blank_line = [blank_line {new_blank}];
end
ds_wsc_dth(end + 1, : ) = blank_line;

%  Compare names to test subset
idx_ssc = arrayfun(@(x) find(ds_ssc.patid == x), cell2mat(cellfun(@(x) str2num(x(5:8)), X_test.names(X_test.cohort_code == 5),'Un',0)));
idx_stages = cellfun(@(x) find(strcmp(ds_stages.s_code, x)), cellfun(@(x) x(1:end-12), X_test.names(X_test.cohort_code == 2), 'Un', 0));
stages_names = cellfun(@(x) x(1:end-12), X_test.names(X_test.cohort_code == 2), 'Un', 0);
idx_stages_old_missing = cellfun(@(x) isempty(find(strcmp(ds_stages_old.s_code, x))), stages_names);
ds_stages_old(end+1:end+sum(idx_stages_old_missing),:) = [stages_names(idx_stages_old_missing), num2cell(nan(sum(idx_stages_old_missing),size(ds_stages_old,2)-1))];
idx_stages_old = cellfun(@(x) find(strcmp(ds_stages_old.s_code, x)), stages_names);
idx_cfs = arrayfun(@(x) find(ds_cfs.nsrrid == x), cell2mat(cellfun(@(x) str2num(x(end-5:end)), X_test.names(X_test.cohort_code == 1),'Un',0)));
idx_mros = cellfun(@(x) find(strcmp(ds_mros.nsrrid, x)), cellfun(@(x) x(end-5:end), X_test.names(X_test.cohort_code == 4), 'Un', 0));
idx_mros_dth = cellfun(@(x) find(strcmp(ds_mros_dth.NSRRID, x)), cellfun(@(x) x(end-5:end), X_test.names(X_test.cohort_code == 4), 'Un', 0));
wsc_names = arrayfun(@(x) [ds_wsc.SUBJ_ID{x} '_' num2str(ds_wsc.VISIT_NUMBER(x))], 1:size(ds_wsc,1),'Un',0);
idx_wsc = cellfun(@(x) find(strcmp(wsc_names' , x)), X_test.names(X_test.cohort_code == 3));
idx_wsc_dth = cellfun(@(x) find(strcmp(ds_wsc_dth.subj_id, x(1:end-2))), X_test.names(X_test.cohort_code == 3),'Un',0);
idx_wsc_dth(cellfun(@(x) isempty(x), idx_wsc_dth)) = {size(ds_wsc_dth,1)};
idx_wsc_dth = cell2mat(idx_wsc_dth);
idx_shhs = arrayfun(@(x) find(ds_shhs.nsrrid == x), cell2mat(cellfun(@(x) str2num(x(end-5:end)), X_test.names(X_test.cohort_code == 6),'Un',0)));
idx_shhs_cvd = arrayfun(@(x) find(ds_shhs_cvd.nsrrid == x), cell2mat(cellfun(@(x) str2num(x(end-5:end)), X_test.names(X_test.cohort_code == 6),'Un',0)));
idx_sof = arrayfun(@(x) find(ds_sof.sofid == x), cell2mat(cellfun(@(x) str2num(x(end-4:end)), X_test.names(X_test.cohort_code == 7),'Un',0)));
idx_hpap = arrayfun(@(x) find(ds_hpap.nsrrid == x), cell2mat(cellfun(@(x) str2num(x(end-6:end)), X_test.names(X_test.cohort_code == 8),'Un',0)));

%% Load ssc stats
ds_ssc_stats_path = "H:\nAge\ssc_stats.xlsx";
ds_ssc_stats_test_extra_path = "H:\nAge\ssc_stats_test_extra.xlsx";
% ds_ssc_stats_test_mortality_path = "H:\nAge\ssc_stats_test_mortality.xlsx";
ds_ssc_stats = readtable(ds_ssc_stats_path);
ds_ssc_stats_test_extra = readtable(ds_ssc_stats_test_extra_path);
% ds_ssc_stats_test_mortality = readtable(ds_ssc_stats_test_mortality_path);
ds_ssc_stats = [ds_ssc_stats; ds_ssc_stats_test_extra];
ds_ssc_stats_names = cellfun(@(x) x(1:end-5), ds_ssc_stats.record, 'Un', 0);
ds_ssc_stats_names_wsc = cellfun(@(x) length(x) == 14, ds_ssc_stats_names);
ds_ssc_stats_names(ds_ssc_stats_names_wsc) = cellfun(@(x) x(1:7), ds_ssc_stats_names(ds_ssc_stats_names_wsc), 'Un', 0);
idx_ssc_stats = cellfun(@(x) find(strcmpi(ds_ssc_stats_names, x)), X_test.names);
ds_ssc_stats = ds_ssc_stats(idx_ssc_stats,:);

%% Load other stats
% Arousal index
ds_ar_pred_path = "C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data\T_AR.txt";
ds_ar_pred = readtable(ds_ar_pred_path);
ds_ar_pred_names = ds_ar_pred.Record_ID;
ds_ar_pred_names_wsc = cellfun(@(x) length(x) == 14, ds_ar_pred_names);
ds_ar_pred_names(ds_ar_pred_names_wsc) = cellfun(@(x) x(1:7), ds_ar_pred_names(ds_ar_pred_names_wsc), 'Un', 0);
ds_ar_pred.Record_ID = ds_ar_pred_names;
idx_ar_pred_missing = cellfun(@(x) isempty(find(strcmpi(ds_ar_pred_names, x), 1)), X_test.names);
ds_ar_pred_names_missing = X_test.names(idx_ar_pred_missing);
ds_ar_pred(end+1:end+size(ds_ar_pred_names_missing,1),:) = [ds_ar_pred_names_missing, num2cell(nan(size(ds_ar_pred_names_missing,1),3))];
idx_ar_pred = cellfun(@(x) find(strcmpi(ds_ar_pred.Record_ID, x)), X_test.names);
ds_ar_pred = ds_ar_pred(idx_ar_pred,:);

ds_sao2_wsc_path = "C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data\WSC_SaO2_80.txt";
ds_sao2_wsc = readtable(ds_sao2_wsc_path);
wsc_names = X_test.names(X_test.cohort_code == 3);
idx_sao2_wsc_missing = cellfun(@(x) isempty(find(strcmpi(ds_sao2_wsc.Record_ID, x), 1)), wsc_names);
ds_sao2_wsc(end+1:end+sum(idx_sao2_wsc_missing),:) = [wsc_names(idx_sao2_wsc_missing), num2cell(nan(sum(idx_sao2_wsc_missing),1))];
idx_sao2_wsc = cellfun(@(x) find(strcmpi(ds_sao2_wsc.Record_ID, x(1:7))), X_test.names(X_test.cohort_code == 3));

%% Collect all PSG and clinical stats
%  Waso (minutes)
waso_ssc = ds_ssc_stats.waso(X_test.cohort_code == 5)/60;%nan(size(idx_ssc)); % ssc missing waso
waso_stages = ds_stages.waso(idx_stages) / 60;
waso_cfs = ds_cfs.waso(idx_cfs);
waso_mros = ds_mros.powaso(idx_mros);
waso_wsc = ds_wsc.waso(idx_wsc);
waso_shhs = ds_shhs.WASO(idx_shhs);
waso_sof = ds_ssc_stats.waso(X_test.cohort_code == 7)/60;
waso_hpap = ds_ssc_stats.waso(X_test.cohort_code == 8)/60;
X_test.waso = [waso_cfs; waso_stages; waso_wsc; waso_mros; waso_ssc; waso_shhs; waso_sof; waso_hpap];

%  SE [0 - 1]
se_ssc = ds_ssc.sleep_efficiency_pct(idx_ssc);
se_stages = ds_stages.se(idx_stages);
se_cfs = ds_cfs.slp_eff(idx_cfs) / 100;
se_mros = ds_mros.poslpeff(idx_mros) / 100;
se_wsc = ds_wsc.sleepeff(idx_wsc);
se_shhs = ds_shhs.slp_eff(idx_shhs) / 100;
se_shhs_2 = ds_ssc_stats.se(X_test.cohort_code == 6);
se_shhs(isnan(se_shhs)) = se_shhs_2(isnan(se_shhs));
se_sof = ds_sof.slp_eff(idx_sof) / 100;
se_hpap = ds_hpap.slpeffp(idx_hpap) / 100;
X_test.se = [se_cfs; se_stages; se_wsc; se_mros; se_ssc; se_shhs; se_sof; se_hpap];

%  SL (minutes)
sl_ssc = nan(size(idx_ssc)); % ssc missing sl
sl_stages = ds_stages.sleep_latency(idx_stages) / 60;
sl_cfs = ds_cfs.slp_lat(idx_cfs);
sl_mros = cellfun(@(x) str2num(x), ds_mros.posllatp(idx_mros), 'Un', 0);
sl_mros(cellfun(@(x) isempty(x), sl_mros)) = {NaN};
sl_mros = cell2mat(sl_mros);
sl_wsc = nan(size(idx_wsc));% wsc missing sl
sl_shhs = ds_shhs.slp_lat(idx_shhs);
sl_sof = ds_sof.slplatp(idx_sof);
sl_hpap = nan(size(idx_hpap));
X_test.sl = [sl_cfs; sl_stages; sl_wsc; sl_mros; sl_ssc; sl_shhs; sl_sof; sl_hpap];

%  TST
tst_ssc = ds_ssc.total_sleep_time_min(idx_ssc);
tst_stages = ds_stages.sleep_time(idx_stages) / 60;
tst_cfs = ds_cfs.slp_time(idx_cfs);
tst_mros = ds_mros.poslprdp(idx_mros);
tst_wsc = ds_wsc.TST(idx_wsc);
tst_shhs = ds_ssc_stats.sleep_time(X_test.cohort_code == 6) / 60;
tst_sof = ds_sof.slpprdp(idx_sof);
tst_hpap = ds_hpap.slpprdp(idx_hpap);
X_test.tst = [tst_cfs; tst_stages; tst_wsc; tst_mros; tst_ssc; tst_shhs; tst_sof; tst_hpap];

%  N1P
n1p_ssc = ds_ssc.stage_1_pct(idx_ssc) * 100;
n1p_stages = ds_stages.n1p(idx_stages) * 100;
n1p_cfs = ds_ssc_stats.n1p(X_test.cohort_code == 1) * 100;
n1p_mros = ds_ssc_stats.n1p(X_test.cohort_code == 4) * 100;
n1p_wsc = 100*ds_ssc_stats.n1p(X_test.cohort_code == 3);%nan(size(idx_wsc)); % wsc missing
n1p_shhs = ds_ssc_stats.n1p(X_test.cohort_code == 6) * 100;
n1p_sof = ds_sof.tmstg1p(idx_sof);
n1p_hpap = 100*ds_ssc_stats.n1p(X_test.cohort_code == 8);
X_test.n1p = [n1p_cfs; n1p_stages; n1p_wsc; n1p_mros; n1p_ssc; n1p_shhs; n1p_sof; n1p_hpap];

%  N2P
n2p_ssc = ds_ssc.stage_2_pct(idx_ssc) * 100;
n2p_stages = ds_stages.n2p(idx_stages) * 100;
n2p_cfs = ds_ssc_stats.n2p(X_test.cohort_code == 1) * 100;
n2p_mros = ds_ssc_stats.n2p(X_test.cohort_code == 4) * 100;
n2p_wsc = 100*ds_ssc_stats.n2p(X_test.cohort_code == 3);%nan(size(idx_wsc)); % wsc missing
n2p_shhs = ds_ssc_stats.n2p(X_test.cohort_code == 6) * 100;
n2p_sof = ds_sof.tmstg2p(idx_sof);
n2p_hpap = 100*ds_ssc_stats.n2p(X_test.cohort_code == 8);
X_test.n2p = [n2p_cfs; n2p_stages; n2p_wsc; n2p_mros; n2p_ssc; n2p_shhs; n2p_sof; n2p_hpap];

%  N3P
n3p_ssc = ds_ssc.stage_3_4_pct(idx_ssc) * 100;
n3p_stages = ds_stages.n3p(idx_stages) * 100;
n3p_cfs = ds_ssc_stats.n3p(X_test.cohort_code == 1) * 100;
n3p_mros = ds_ssc_stats.n3p(X_test.cohort_code == 4) * 100;
n3p_wsc = 100*ds_ssc_stats.n3p(X_test.cohort_code == 3);%nan(size(idx_wsc)); % wsc missing
n3p_shhs = ds_ssc_stats.n3p(X_test.cohort_code == 6) * 100;
n3p_sof = ds_sof.tmstg34p(idx_sof);
n3p_hpap = 100*ds_ssc_stats.n3p(X_test.cohort_code == 8);
X_test.n3p = [n3p_cfs; n3p_stages; n3p_wsc; n3p_mros; n3p_ssc; n3p_shhs; n3p_sof; n3p_hpap];

%  REMP
remp_ssc = ds_ssc.stage_rem_pct(idx_ssc) * 100;
remp_stages = ds_stages.remp(idx_stages) * 100;
remp_cfs = ds_ssc_stats.remp(X_test.cohort_code == 1) * 100;
remp_mros = ds_ssc_stats.remp(X_test.cohort_code == 4) * 100;
remp_wsc = 100*ds_ssc_stats.remp(X_test.cohort_code == 3);%nan(size(idx_wsc));
remp_shhs = ds_ssc_stats.remp(X_test.cohort_code == 6) * 100;
remp_sof = ds_sof.tmremp(idx_sof);
remp_hpap = 100*ds_ssc_stats.remp(X_test.cohort_code == 8);
X_test.remp = [remp_cfs; remp_stages; remp_wsc; remp_mros; remp_ssc; remp_shhs; remp_sof; remp_hpap];

%  ArI (Run arousal detector)
ari_ssc = ds_ar_pred.ARI(X_test.cohort_code == 5);%nan(size(idx_ssc));
ari_stages = nan(size(idx_stages));
ari_cfs = ds_cfs.ai_all(idx_cfs);
ari_mros = cellfun(@(x) str2num(x), ds_mros.poai_all(idx_mros), 'Un', 0);
ari_mros(cellfun(@(x) isempty(x), ari_mros)) = {NaN};
ari_mros = cell2mat(ari_mros);
ari_mros_2 = ds_ar_pred.ARI(X_test.cohort_code == 4);
ari_mros(isnan(ari_mros)) = ari_mros_2(isnan(ari_mros));
ari_wsc = ds_ar_pred.ARI(X_test.cohort_code == 3);%nan(size(idx_wsc)); % Get from MAD
ari_shhs = ds_shhs.ai_all(idx_shhs);
ari_shhs_2 = ds_ar_pred.ARI(X_test.cohort_code == 6);
ari_shhs(isnan(ari_shhs)) = ari_shhs_2(isnan(ari_shhs));
ari_sof = ds_sof.ai_all(idx_sof);
ari_hpap = nan(size(idx_hpap));
X_test.ari = [ari_cfs; ari_stages; ari_wsc; ari_mros; ari_ssc; ari_shhs; ari_sof; ari_hpap];

%  BMI
bmi_ssc = ds_ssc.bmi(idx_ssc);
bmi_stages = ds_stages_old.bmi(idx_stages_old);
bmi_cfs = ds_cfs.bmi(idx_cfs);
bmi_mros = cellfun(@(x) str2num(x), ds_mros.hwbmi(idx_mros));
bmi_wsc = ds_wsc.bmi(idx_wsc);
bmi_shhs = ds_shhs.bmi_s1(idx_shhs);
bmi_sof = ds_sof.V8BMI(idx_sof);
bmi_hpap = ds_hpap.bmi(idx_hpap);
X_test.bmi = [bmi_cfs; bmi_stages; bmi_wsc; bmi_mros; bmi_ssc; bmi_shhs; bmi_sof; bmi_hpap];

%  Mortality
vital_ssc = nan(size(idx_ssc));
vital_stages = nan(size(idx_stages));
vital_cfs = nan(size(idx_cfs));
vital_mros = cellfun(@(x) strcmp('0',x)*(1) + strcmp('1',x)*0 + strcmp('.',x)*(-1), ds_mros_dth.DADEAD(idx_mros_dth));
vital_mros(vital_mros == -1) = nan;
vital_wsc = 1 - ds_wsc_dth.DADEAD(idx_wsc_dth);
vital_shhs = ds_shhs_cvd.vital(idx_shhs_cvd);
vital_sof = cellfun(@(x) strcmp('0',x)*(1) + strcmp('D',x)*(0) + strcmp('1',x)*0 + strcmp('',x)*(-1), ds_sof.V8DTHASH(idx_sof));
vital_sof(vital_sof == -1) = nan;
vital_hpap = nan(size(idx_hpap));
X_test.vital = [vital_cfs; vital_stages; vital_wsc; vital_mros; vital_ssc; vital_shhs; vital_sof; vital_hpap];

%  Mortality Cardiovascular
vitalC_ssc = nan(size(idx_ssc));
vitalC_stages = nan(size(idx_stages));
vitalC_cfs = nan(size(idx_cfs));
vitalC_mros = cellfun(@(x) strcmp('0',x)*(1) + strcmp('1',x)*0 + strcmp('.',x)*(-1), ds_mros_dth.DACARDIO(idx_mros_dth));
vitalC_mros(vitalC_mros == -1) = nan;
vitalC_wsc = cellfun(@(x) strcmp('NA',x)*(-1) + strcmp('0',x)*1 + strcmp('',x)*(-1), ds_wsc_dth.DACARDIO(idx_wsc_dth));
vitalC_wsc(vitalC_wsc == -1) = nan;
vitalC_shhs = 1 - ds_shhs_cvd.cvd_death(idx_shhs_cvd);
vitalC_shhs(isnan(vitalC_shhs) & vital_shhs == 1) = 1;
vitalC_sof = arrayfun(@(x) ~(strcmp(ds_sof.V8DTHASH(x), '1') | strcmp(ds_sof.V8DTHCHD(x), '1') | strcmp(ds_sof.V8DTHSTK(x), '1') | strcmp(ds_sof.V8DTHSUD(x), '1')), idx_sof);
vitalC_hpap = nan(size(idx_hpap));
X_test.vitalC = [vitalC_cfs; vitalC_stages; vitalC_wsc; vitalC_mros; vitalC_ssc; vitalC_shhs; vitalC_sof; vitalC_hpap];

%  Mortality Cancer (missing for SHHS)
% vitalCC_ssc = nan(size(idx_ssc));
% vitalCC_stages = nan(size(idx_stages));
% vitalCC_cfs = nan(size(idx_cfs));
% vitalCC_mros = cellfun(@(x) strcmp('0',x)*(1) + strcmp('1',x)*0 + strcmp('.',x)*(-1), ds_mros_dth.DACANCER(idx_mros_dth));
% vitalCC_mros(vitalC_mros == -1) = nan;
% vitalCC_wsc = cellfun(@(x) strcmp('NA',x)*(-1) + strcmp('0',x)*1 + strcmp('',x)*(-1), ds_wsc_dth.DACANCER(idx_wsc_dth));
% vitalCC_wsc(vitalC_wsc == -1) = nan;
% vitalCC_shhs = 1 - ds_shhs_cvd.cvd_death(idx_shhs_cvd);
% vitalCC_sof = nan(size(idx_sof));
% vitalCC_hpap = nan(size(idx_hpap));
% X_test.vitalCC = [vitalCC_cfs; vitalCC_stages; vitalCC_wsc; vitalCC_mros; vitalCC_ssc; vitalCC_shhs; vitalCC_sof; vitalCC_hpap];

%  Time to death
t_death_ssc = nan(size(idx_ssc));
t_death_stages = nan(size(idx_stages));
t_death_cfs = nan(size(idx_cfs));
t_death_mros = cellfun(@(x) str2num(x), ds_mros_dth.FUVSDT(idx_mros_dth));
t_death_wsc = 365*(min(ds_wsc_dth.Censored(idx_wsc_dth), ds_wsc_dth.FUTIME(idx_wsc_dth)) - (X_test.age(X_test.cohort_code == 3) - ds_wsc_dth.ageatvisit(idx_wsc_dth)));
t_death_shhs = ds_shhs_cvd.censdate(idx_shhs_cvd);
t_death_sof = ds_sof.V8FOLALL(idx_sof);
t_death_hpap = nan(size(idx_hpap));
X_test.t_death = [t_death_cfs; t_death_stages; t_death_wsc; t_death_mros; t_death_ssc; t_death_shhs; t_death_sof; t_death_hpap];

%  Smoking
smoke_ssc = nan(size(idx_ssc));
smoke_stages = nan(size(idx_stages));
smoke_cfs = cellfun(@(x) strcmp('current',x)*2 + strcmp('quit',x)*1, ds_cfs.smoker(idx_cfs));
smoke_mros = cellfun(@(x) strcmp('2',x)*2 + strcmp('1',x)*1 + strcmp('A',x)*(-1), ds_mros.tursmoke(idx_mros));
smoke_mros(smoke_mros == -1) = nan;
smoke_wsc = cellfun(@(x) strcmp('C',x)*2 + strcmp('P',x)*1 + strcmp('',x)*(-1), ds_wsc_dth.smoker(idx_wsc_dth));
smoke_wsc(smoke_wsc == -1) = nan;
smoke_shhs = ds_shhs.smokstat_s1(idx_shhs)*2;
smoke_shhs(smoke_shhs == 4) = 1;
smoke_shhs(isnan(smoke_shhs)) = nan;
smoke_sof = nan(size(idx_sof));
smoke_hpap = nan(size(idx_hpap));
X_test.smoke = [smoke_cfs; smoke_stages; smoke_wsc; smoke_mros; smoke_ssc; smoke_shhs; smoke_sof; smoke_hpap];
X_test.smoke_02 = double(X_test.smoke == 2);
X_test.smoke_02(isnan(X_test.smoke)) = nan;
X_test.smoke_01 = double(X_test.smoke == 1);
X_test.smoke_01(isnan(X_test.smoke)) = nan;

%  Education (NOTE EDUCATION IS SEPARATE FOR EACH COHORT) -> reformat
% 1: Less than 10 years of education
% 2: 11-15 years of education
% 3: 16-20 years of education
% 4: More than 20 years of education
edu_ssc = nan(size(idx_ssc));
edu_stages = nan(size(idx_stages));
edu_cfs = nan(size(idx_cfs));
edu_mros = ds_mros.gieduc(idx_mros);   % education classifications [(1: 10>), (2: 10>), (3: 11-15), (4: 11-15), (5: 16-20), (6: 16-20), (7: 16-20), (8: >20)]
edu_mros(ds_mros.gieduc(idx_mros) == 1) = 1;
edu_mros(ds_mros.gieduc(idx_mros) == 2) = 1;
edu_mros(ds_mros.gieduc(idx_mros) == 3) = 2;
edu_mros(ds_mros.gieduc(idx_mros) == 4) = 2;
edu_mros(ds_mros.gieduc(idx_mros) == 5) = 3;
edu_mros(ds_mros.gieduc(idx_mros) == 6) = 3;
edu_mros(ds_mros.gieduc(idx_mros) == 7) = 3;
edu_mros(ds_mros.gieduc(idx_mros) == 8) = 4;
edu_wsc = ds_wsc_dth.EDU(idx_wsc_dth); % Education years
edu_wsc(ds_wsc_dth.EDU(idx_wsc_dth) < 10) = 1;
edu_wsc(ds_wsc_dth.EDU(idx_wsc_dth) >= 10) = 2;
edu_wsc(ds_wsc_dth.EDU(idx_wsc_dth) >= 16) = 3;
edu_wsc(ds_wsc_dth.EDU(idx_wsc_dth) >= 20) = 4;
edu_shhs = ds_shhs.educat(idx_shhs);   % education years classifications
edu_sof = nan(size(idx_sof));
edu_hpap = nan(size(idx_hpap));
X_test.edu = [edu_cfs; edu_stages; edu_wsc; edu_mros; edu_ssc; edu_shhs; edu_sof; edu_hpap];
X_test.edu_01 = double(X_test.edu == 1);
X_test.edu_01(isnan(X_test.edu)) = nan;
X_test.edu_02 = double(X_test.edu == 2);
X_test.edu_02(isnan(X_test.edu)) = nan;
X_test.edu_03 = double(X_test.edu == 3);
X_test.edu_03(isnan(X_test.edu)) = nan;
X_test.edu_04 = double(X_test.edu == 4);
X_test.edu_04(isnan(X_test.edu)) = nan;

%  Race (1: white, 2: black, 3: other)
race_ssc = nan(size(idx_ssc));
race_stages = nan(size(idx_stages));
race_cfs = ds_cfs.race(idx_cfs);
race_mros = ds_mros.girace(idx_mros);
race_mros(race_mros > 2) = 3;
race_wsc = cellfun(@(x) strcmp('B',x)*2 + strcmp('W',x)*1 + strcmp('A',x)*(3) + strcmp('H',x)*(3) + strcmp('I',x)*(3) + strcmp('O',x)*(3) + strcmp('',x)*(-1), ds_wsc_dth.RACE(idx_wsc_dth));
race_wsc(race_wsc == -1) = nan;
race_shhs = ds_shhs.race(idx_shhs);
race_sof = ds_sof.race(idx_sof);
race_hpap = ds_hpap.race3(idx_hpap);
X_test.race = [race_cfs; race_stages; race_wsc; race_mros; race_ssc; race_shhs; race_sof; race_hpap];
X_test.race_01 = double(X_test.race == 1);
X_test.race_01(isnan(X_test.race)) = nan;

%  Marital Status {1: married, 2: single, 3: separated/divorced, 4: widow}
marital_ssc = nan(size(idx_ssc));
marital_stages = nan(size(idx_stages));
marital_cfs = ds_cfs.MARSTAT(idx_cfs);
marital_mros_temp = ds_mros.gimstat(idx_mros);
marital_mros = marital_mros_temp;
marital_mros(marital_mros_temp == 5) = 2;
marital_mros(marital_mros_temp == 4) = 3;
marital_mros(marital_mros_temp == 2) = 4;
marital_wsc = nan(size(idx_wsc));
marital_shhs_temp = ds_shhs.MStat(idx_shhs);
marital_shhs = marital_shhs_temp;
marital_shhs(marital_shhs_temp == 2) = 4;
marital_shhs(marital_shhs_temp == 4) = 2;
marital_shhs(marital_shhs_temp == 8) = nan;
marital_sof = nan(size(idx_sof));
marital_hpap = nan(size(idx_hpap));
X_test.marital = [marital_cfs; marital_stages; marital_wsc; marital_mros; marital_ssc; marital_shhs; marital_sof; marital_hpap];
X_test.marital_01 = double(X_test.marital == 1);
X_test.marital_01(isnan(X_test.marital)) = nan;

%  Alcohol Use (drinks / day)
alch_ssc = nan(size(idx_ssc));
alch_stages = nan(size(idx_stages));
alch_cfs = nan(size(idx_cfs));
alch_mros = cellfun(@(x) strcmp('A',x)*(-1) + strcmp('K',x)*(-1) + strcmp('0',x)*(0) + strcmp('1',x)*(0.5) + strcmp('2',x)*(1.5) + strcmp('3',x)*(4) + strcmp('4',x)*(9.5) + strcmp('5',x)*(14), ds_mros.tudramt(idx_mros));
alch_mros(alch_mros == -1) = nan;
alch_mros = alch_mros / 7;
alch_wsc = ds_wsc.BEER_WEEK(idx_wsc) / 7;
alch_shhs = ds_shhs.Alcoh(idx_shhs);
alch_sof = ds_sof.V1DRWKA(idx_sof) / 7;
alch_hpap = nan(size(idx_hpap));
X_test.alch = [alch_cfs; alch_stages; alch_wsc; alch_mros; alch_ssc; alch_shhs; alch_sof; alch_hpap];

% Caffiene use (daily intake)  (MROS is for the day before)
caf_ssc = nan(size(idx_ssc));
caf_stages = nan(size(idx_stages));
caf_cfs = nan(size(idx_cfs));
caf_mros_cof = cellfun(@(x) str2num(x), ds_mros.poxcoff(idx_mros), 'Un',0);
caf_mros_cof(cellfun(@isempty, caf_mros_cof)) = {nan};
caf_mros_sod = cellfun(@(x) str2num(x), ds_mros.poxsoda(idx_mros), 'Un',0);
caf_mros_sod(cellfun(@isempty, caf_mros_sod)) = {nan};
caf_mros_tea = cellfun(@(x) str2num(x), ds_mros.poxtea(idx_mros), 'Un',0);
caf_mros_tea(cellfun(@isempty, caf_mros_tea)) = {nan};
caf_mros = cell2mat(caf_mros_cof) + cell2mat(caf_mros_sod) + cell2mat(caf_mros_tea);
caf_wsc = ds_wsc.CANS_COLA(idx_wsc) + ds_wsc.CUPS_COFFEE(idx_wsc);
caf_shhs = ds_shhs.COFFEE15(idx_shhs) + ds_shhs.SODA15(idx_shhs) + ds_shhs.TEA15(idx_shhs);
caf_sof = nan(size(idx_sof));
caf_hpap = nan(size(idx_hpap));
X_test.caf = [caf_cfs; caf_stages; caf_wsc; caf_mros; caf_ssc; caf_shhs; caf_sof; caf_hpap];

% Antidepressants (how to group, ask eileen and mignot)
ant_ssc = nan(size(idx_ssc)); % Medication is string format and could be grouped
ant_stages = nan(size(idx_stages));
ant_cfs = double((ds_cfs.ANTIDEPR(idx_cfs) + ds_cfs.OTHANTID(idx_cfs)) > 0);
ant_mros = ds_mros.m1adepr(idx_mros);
ant_wsc = ds_wsc.antidep(idx_wsc);
ant_shhs = double((ds_shhs.NTCA1(idx_shhs) + ds_shhs.TCA1(idx_shhs)) > 0);
ant_sof = nan(size(idx_sof));
ant_hpap = nan(size(idx_hpap));
X_test.ant = [ant_cfs; ant_stages; ant_wsc; ant_mros; ant_ssc; ant_shhs; ant_sof; ant_hpap];

% Benzodiazepines
ben_ssc = nan(size(idx_ssc)); % Medication is string format and could be grouped
ben_stages = nan(size(idx_stages));
ben_cfs = nan(size(idx_cfs));
ben_mros = ds_mros.m1benzo(idx_mros);
ben_wsc = ds_wsc.bd_drug(idx_wsc); % is this correct?
ben_shhs = ds_shhs.BENZOD1(idx_shhs);
ben_sof = nan(size(idx_sof));
ben_hpap = nan(size(idx_hpap));
X_test.ben = [ben_cfs; ben_stages; ben_wsc; ben_mros; ben_ssc; ben_shhs; ben_sof; ben_hpap];

% Sleep medication (what is this?)
sme_ssc = nan(size(idx_ssc));
sme_stages = nan(size(idx_stages));
sme_cfs = ds_cfs.SLPPL3DY(idx_cfs);
sme_mros = double((ds_mros.m1zolp(idx_mros) + ds_mros.m1zolpsg(idx_mros) + ds_mros.m1slpmed(idx_mros) + ds_mros.m1nbanx(idx_mros)) > 0);
sme_wsc = ds_wsc.sedative_drug(idx_wsc);
sme_shhs = nan(size(idx_shhs));
sme_sof = nan(size(idx_sof));
sme_hpap = nan(size(idx_hpap));
X_test.sme = [sme_cfs; sme_stages; sme_wsc; sme_mros; sme_ssc; sme_shhs; sme_sof; sme_hpap];

% Sleep time with 80%> SaO2
O280_ssc = nan(size(idx_ssc));
O280_stages = nan(size(idx_stages));
O280_cfs = ds_cfs.PER80(idx_cfs) .* tst_cfs / 100;
O280_mros = ds_mros.popcsa80(idx_mros) .* tst_mros / 100;
O280_wsc = ds_sao2_wsc.SaO2_80(idx_sao2_wsc)/60;%nan(size(idx_wsc)); % could be calculated
O280_shhs = ds_shhs.pctlt80(idx_shhs) .* tst_shhs / 100;
O280_sof = nan(size(idx_sof));
O280_hpap = nan(size(idx_hpap));
X_test.O280 = [O280_cfs; O280_stages; O280_wsc; O280_mros; O280_ssc; O280_shhs; O280_sof; O280_hpap];

% Sleep time with 80%> SaO2
% O280_ssc = nan(size(idx_ssc));
% O280_stages = nan(size(idx_stages));
% O280_cfs = ds_cfs.PER80(idx_cfs) .* tst_cfs;
% O280_mros = ds_mros.popcsa80(idx_mros) .* tst_mros;
% O280_wsc = nan(size(idx_sao2_wsc)); % could be calculated
% O280_shhs = ds_shhs.pctlt80(idx_shhs) .* tst_shhs;
% O280_sof = nan(size(idx_sof));
% O280_hpap = nan(size(idx_hpap));
% X_test.O280 = [O280_cfs; O280_stages; O280_wsc; O280_mros; O280_ssc; O280_shhs; O280_sof; O280_hpap];

% Eppworth Sleepiness Score
ESS_ssc = ds_ssc.ess(idx_ssc);
ESS_stages = nan(size(idx_stages));
ESS_cfs = ds_cfs.ESSSCOR(idx_cfs);
ESS_mros = ds_mros.epepwort(idx_mros);
ESS_wsc = ds_wsc.doze_sc(idx_wsc);
ESS_shhs = ds_shhs.ESS_s1(idx_shhs);
ESS_sof = nan(size(idx_sof));
ESS_hpap = ds_hpap.esstotal(idx_hpap);
X_test.ESS = [ESS_cfs; ESS_stages; ESS_wsc; ESS_mros; ESS_ssc; ESS_shhs; ESS_sof; ESS_hpap];

% Teng Mini Mental State Examination Score (missing for now)

% Depression
dep_ssc = nan(size(idx_ssc));
dep_stages = nan(size(idx_stages));
dep_cfs = ds_cfs.DEPDIAG(idx_cfs);
dep_cfs(dep_cfs == -2) = nan;
dep_mros = nan(size(idx_mros)); % missing
dep_wsc = ds_wsc.depressed(idx_wsc);
dep_shhs = nan(size(idx_shhs));
dep_sof = cellfun(@(x) strcmp(x, '0')*0 + strcmp(x, '1')*1 + strcmp(x, '')*(-1) + strcmp(x, 'O')*(-1), ds_sof.V6SDEPR(idx_sof));
dep_sof(dep_sof == -1) = nan;
dep_hpap = nan(size(idx_hpap));
X_test.dep = [dep_cfs; dep_stages; dep_wsc; dep_mros; dep_ssc; dep_shhs; dep_sof; dep_hpap];

% Congestive Heart Failure
chf_ssc = nan(size(idx_ssc));
chf_stages = nan(size(idx_stages));
chf_cfs = nan(size(idx_cfs));
chf_mros =  cellfun(@(x) strcmp('A',x)*(-1) + strcmp('0',x)*(0) + strcmp('1',x)*(1), ds_mros.mhchf(idx_mros)); % chf, chronic obstructive lung disease, or emphysema?
chf_mros(chf_mros == -1) = nan;
chf_wsc = cellfun(@(x) strcmp('',x)*(-1) + strcmp('N',x)*(0) + strcmp('Y',x)*(1), ds_wsc.CONGESTIVEHF_YND(idx_wsc));
chf_wsc(chf_wsc == -1) = nan;
chf_shhs = double(ds_shhs_cvd.prev_chf(idx_shhs_cvd) > 0);
cfs_shhs(isnan(ds_shhs_cvd.prev_chf(idx_shhs_cvd))) = nan;
chf_sof = ds_sof.V8ECONG(idx_sof);
chf_hpap = nan(size(idx_hpap));
X_test.chf = [chf_cfs; chf_stages; chf_wsc; chf_mros; chf_ssc; chf_shhs; chf_sof; chf_hpap];

% Chronic Obstructive Pulmonary Disease
copd_ssc = nan(size(idx_ssc));
copd_stages = nan(size(idx_stages));
copd_cfs = nan(size(idx_cfs));
copd_mros =  cellfun(@(x) strcmp('A',x)*(-1) + strcmp('0',x)*(0) + strcmp('1',x)*(1), ds_mros.mhcobpd(idx_mros)); % COPD, chronic obstructive lung disease, or emphysema?
copd_mros(copd_mros == -1) = nan;
copd_wsc = nan(size(idx_wsc));
copd_shhs = ds_shhs.COPD15(idx_shhs);
copd_shhs(copd_shhs == 8) = nan;
copd_sof = ds_sof.V8ECOPD(idx_sof);
copd_hpap = nan(size(idx_hpap));
X_test.copd = [copd_cfs; copd_stages; copd_wsc; copd_mros; copd_ssc; copd_shhs; copd_sof; copd_hpap];


% Type 2 diabetes
t2d_ssc = nan(size(idx_ssc));
t2d_stages = nan(size(idx_stages));
t2d_cfs = ds_cfs.diabetes2(idx_cfs);
t2d_mros = cellfun(@(x) strcmp('A',x)*(-1) + strcmp('0',x)*(0) + strcmp('1',x)*(1), ds_mros.mhdiab(idx_mros));
t2d_mros(t2d_mros == -1) = nan;
t2d_wsc = cellfun(@(x) strcmp('',x)*(-1) + strcmp('N',x)*(0) + strcmp('Y',x)*(1), ds_wsc_dth.DIABETES_YND(idx_wsc_dth));
t2d_wsc(t2d_wsc == -1) = nan;
t2d_shhs = ds_shhs.ParRptDiab(idx_shhs);
t2d_sof = ds_sof.V8EDIAB(idx_sof);
t2d_hpap = nan(size(idx_hpap));
X_test.t2d = [t2d_cfs; t2d_stages; t2d_wsc; t2d_mros; t2d_ssc; t2d_shhs; t2d_sof; t2d_hpap];

% Heart attack
haa_ssc = nan(size(idx_ssc));
haa_stages = nan(size(idx_stages));
haa_cfs = ds_cfs.HRTDIAG(idx_cfs);
haa_cfs(haa_cfs == -2) = nan;
haa_mros = cellfun(@(x) strcmp('A',x)*(-1) + strcmp('0',x)*(0) + strcmp('1',x)*(1), ds_mros.mhmi(idx_mros));
haa_mros(haa_mros == -1) = nan;
haa_wsc = cellfun(@(x) strcmp('',x)*(-1) + strcmp('N',x)*(0) + strcmp('Y',x)*(1), ds_wsc.HEARTATTACK_YND(idx_wsc));
haa_wsc(haa_wsc == -1) = nan;
haa_shhs = double(ds_shhs_cvd.prev_mi(idx_shhs_cvd) > 0);
haa_shhs(isnan(ds_shhs_cvd.prev_mi(idx_shhs_cvd))) = nan;
haa_sof = ds_sof.V8EHEART(idx_sof);
haa_hpap = nan(size(idx_hpap));
X_test.haa = [haa_cfs; haa_stages; haa_wsc; haa_mros; haa_ssc; haa_shhs; haa_sof; haa_hpap];

% Stroke
stro_ssc = nan(size(idx_ssc));
stro_stages = nan(size(idx_stages));
stro_cfs = ds_cfs.STRODIAG(idx_cfs);
stro_mros = cellfun(@(x) strcmp('A',x)*(-1) + strcmp('0',x)*(0) + strcmp('1',x)*(1), ds_mros.mhstrk(idx_mros));
stro_mros(stro_mros == -1) = nan;
stro_wsc = cellfun(@(x) strcmp('',x)*(-1) + strcmp('N',x)*(0) + strcmp('Y',x)*(1), ds_wsc.STROKE_YND(idx_wsc));
stro_wsc(stro_wsc == -1) = nan;
stro_shhs = double(ds_shhs_cvd.prev_stk(idx_shhs_cvd) > 0);
stro_shhs(isnan(ds_shhs_cvd.prev_stk(idx_shhs_cvd))) = nan;
stro_sof = ds_sof.V8ESTRK(idx_sof);
stro_hpap = nan(size(idx_hpap));
X_test.stro = [stro_cfs; stro_stages; stro_wsc; stro_mros; stro_ssc; stro_shhs; stro_sof; stro_hpap];

% Hypertension (TODO: Implement this)
hype_ssc = ds_ssc.hypertension_bool(idx_ssc);
hype_stages = nan(size(idx_stages));
hype_cfs = double(ds_cfs.htndx(idx_cfs) + ds_cfs.htnx(idx_cfs) > 0);
hype_mros = ds_mros_dth.MHBP(idx_mros_dth);
hype_mros(cellfun(@(x) strcmp(x, '.'), hype_mros)) = {'-10'};
hype_mros = cellfun(@(x) str2num(strrep(x,',','.')), hype_mros);
hype_mros(hype_mros == -10) = NaN;
hype_wsc = cellfun(@(x) strcmp('',x)*(-1) + strcmp('N',x)*(0) + strcmp('Y',x)*(1), ds_wsc.HYPERTENSION_YND(idx_wsc));
hype_wsc(hype_wsc == -1) = nan;
hype_shhs = ds_shhs.HTNDerv_s1(idx_shhs);
hype_sof = ds_sof.V8EHYPER(idx_sof);
hype_hpap = nan(size(idx_hpap));
X_test.hype = [hype_cfs; hype_stages; hype_wsc; hype_mros; hype_ssc; hype_shhs; hype_sof; hype_hpap];

% Actigraphy Mean Scored Sleep While Outside of Sleep Interval
ams_ssc = nan(size(idx_ssc));
ams_stages = nan(size(idx_stages));
ams_cfs = nan(size(idx_cfs));
ams_mros = ds_mros_dth.ACSMINMP(idx_mros_dth);
ams_mros(cellfun(@(x) strcmp(x, '.'), ams_mros)) = {'-10'};
ams_mros = cellfun(@(x) str2num(strrep(x,',','.')), ams_mros);
ams_mros(ams_mros == -10) = NaN;
ams_wsc = nan(size(idx_wsc));
ams_shhs = nan(size(idx_shhs));
ams_sof = nan(size(idx_sof));
ams_hpap = nan(size(idx_hpap));
X_test.ams = [ams_cfs; ams_stages; ams_wsc; ams_mros; ams_ssc; ams_shhs; ams_sof; ams_hpap];

% Actigraphy Wake After Sleep Onset
awaso_ssc = nan(size(idx_ssc));
awaso_stages = nan(size(idx_stages));
awaso_cfs = nan(size(idx_cfs));
awaso_mros = ds_mros_dth.ACWASOMP(idx_mros_dth);
awaso_mros(cellfun(@(x) strcmp(x, '.'), awaso_mros)) = {'-10'};
awaso_mros = cellfun(@(x) str2num(strrep(x,',','.')), awaso_mros);
awaso_mros(awaso_mros == -10) = NaN;
awaso_wsc = nan(size(idx_wsc));
awaso_shhs = nan(size(idx_shhs));
awaso_sof = nan(size(idx_sof));
awaso_hpap = nan(size(idx_hpap));
X_test.awaso = [awaso_cfs; awaso_stages; awaso_wsc; awaso_mros; awaso_ssc; awaso_shhs; awaso_sof; awaso_hpap];

% Teng Mini Mental State Examination Score
mmse_ssc = nan(size(idx_ssc));
mmse_stages = nan(size(idx_stages));
mmse_cfs = nan(size(idx_cfs));
mmse_mros = ds_mros_dth.TMMSCORE(idx_mros_dth);
mmse_mros(cellfun(@(x) strcmp(x, '.'), mmse_mros)) = {'-10'};
mmse_mros = cellfun(@(x) str2num(strrep(x,',','.')), mmse_mros);
mmse_mros(mmse_mros == -10) = NaN;
mmse_wsc = nan(size(idx_wsc));
mmse_shhs = nan(size(idx_shhs));
mmse_sof = nan(size(idx_sof));
mmse_hpap = nan(size(idx_hpap));
X_test.mmse = [mmse_cfs; mmse_stages; mmse_wsc; mmse_mros; mmse_ssc; mmse_shhs; mmse_sof; mmse_hpap];

% Physical Activity Scale for the Elderly Score,
pase_ssc = nan(size(idx_ssc));
pase_stages = nan(size(idx_stages));
pase_cfs = nan(size(idx_cfs));
pase_mros = ds_mros_dth.PASCORE(idx_mros_dth);
pase_mros(cellfun(@(x) strcmp(x, '.'), pase_mros)) = {'-10'};
pase_mros = cellfun(@(x) str2num(strrep(x,',','.')), pase_mros);
pase_mros(pase_mros == -10) = NaN;
pase_wsc = nan(size(idx_wsc));
pase_shhs = nan(size(idx_shhs));
pase_sof = nan(size(idx_sof));
pase_hpap = nan(size(idx_hpap));
X_test.pase = [pase_cfs; pase_stages; pase_wsc; pase_mros; pase_ssc; pase_shhs; pase_sof; pase_hpap];

% Visit 1? (WSC has multiple visits)
idx_first_visit_all = zeros(size(X_test.age));
unique_subj_id = unique(cellfun(@(x) x(1:end-2), X_test.names(X_test.cohort_code == 3),'Un',0));
subj_id_short = cellfun(@(x) x(1:end-2), X_test.names, 'Un', 0);
for i = 1:length(unique_subj_id)
    idx_id = find(strcmp(subj_id_short, unique_subj_id{i}));
    idx_id = idx_id(X_test.age(idx_id) == min(X_test.age(idx_id)));
    idx_first_visit_all(idx_id) = 1;
end
v1_wsc = X_test.cohort_code == 3 & idx_first_visit_all; % & ~idx_in_train;
v1_all = ones(size(X_test.age));
v1_all(X_test.cohort_code == 3) = v1_wsc(X_test.cohort_code == 3);
X_test.v1 = v1_all;

%% Exclude training and validation data that has missing predictions
train_val_predicted = dir('H:\nAge\test_mortality_F_eegeogemg5\*.hdf5');
train_val_predicted = {train_val_predicted.name};
idx_train_val = find(X_test.mode ~= 2);
idx_train_val_remove = cellfun(@(x) ~any(contains(train_val_predicted, x, 'IgnoreCase', true)), strrep(X_test.names(idx_train_val),'_EDFAndScore',''));
X_test(idx_train_val(idx_train_val_remove),:) = [];

%% Read Data
%  Prediction data
paths = {'H:\nAge\model_1','H:\nAge\model_10','H:\nAge\model_5', ...
    'H:\nAge\model_eeg5', 'H:\nAge\model_eegeogemg5','H:\nAge\model_ecg5', ...
    'H:\nAge\model_resp5','H:\nAge\model_com5'};
mergestructs = @(x,y) cell2struct([struct2cell(x);struct2cell(y)],[fieldnames(x);fieldnames(y)]);
names = {'1','10','5','eeg','ssc','ecg','resp','comb'};

for i = 1:length(paths)
    [preds, ~, metrics_pre] = read_model_metrics(paths{i});
    [preds_test_extra, ~, metrics_pre_test_extra] = read_model_metrics(paths{i},'_test_ext');
    [preds_test_mortality, ~, metrics_pre_test_mortality] = read_model_metrics(paths{i},'_test_mortality');
    preds = mergestructs(preds, preds_test_extra);
    preds = mergestructs(preds, preds_test_mortality);
    metrics_pre = [metrics_pre; metrics_pre_test_extra; metrics_pre_test_mortality];
    [SA, SA_pre] = match_sleep_age(preds, metrics_pre, X_test, '_EDFAndScore');
    X_test.(['SA_' names{i}]) = SA;
    X_test.(['SAI_' names{i}]) = SA - X_test.age;
    X_test.(['SA_pre_' names{i}]) = SA_pre;
    X_test.(['SAI_pre_' names{i}]) = SA_pre - X_test.age;
end

%% Exclude all >= 90 years ???
X_test(X_test.age >= 90 & X_test.mode == 2,:) = [];
X_test(X_test.age < 20 & X_test.mode == 2,:) = [];

%% Add mean combined model
names = [names {'comb_avg'}];
SA = (X_test.(['SA_' names{4}]) + X_test.(['SA_' names{5}]) + X_test.(['SA_' names{6}]) + X_test.(['SA_' names{7}]))/4;
SA_pre = (X_test.(['SA_pre_' names{4}]) + X_test.(['SA_pre_' names{5}]) + X_test.(['SA_' names{6}]) + X_test.(['SA_pre_' names{7}]))/4;
X_test.(['SA_' names{end}]) = SA;
X_test.(['SAI_' names{end}]) = SA - X_test.age;
X_test.(['SA_pre_' names{end}]) = SA_pre;
X_test.(['SAI_pre_' names{end}]) = SA_pre - X_test.age;

names = [names {'comb_eeg'}];
SA = (X_test.(['SA_' names{4}]) + X_test.(['SA_' names{5}]))/2;
SA_pre = (X_test.(['SA_pre_' names{4}]) + X_test.(['SA_pre_' names{5}]))/2;
X_test.(['SA_' names{end}]) = SA;
X_test.(['SAI_' names{end}]) = SA - X_test.age;
X_test.(['SA_pre_' names{end}]) = SA_pre;
X_test.(['SAI_pre_' names{end}]) = SA_pre - X_test.age;

%% Add a simple regression model
names = [names {'basic'}];
in_c = 1:6;
idx_data = ismember(X_test.cohort_code, in_c) & ismember(X_test.mode, 0);
idx_pv = find(ismember(X_test.Properties.VariableNames,{'ahi','tst','waso','n1p','n2p','n3p','remp'}));
idx_rv = find(ismember(X_test.Properties.VariableNames,{'age'}));
mdl = fitlm(X_test(idx_data,:),'ResponseVar',idx_rv,'PredictorVars',idx_pv);
SA = predict(mdl, X_test);
X_test.(['SA_' names{end}]) = SA;
X_test.(['SAI_' names{end}]) = SA - X_test.age;
X_test.(['SA_pre_' names{end}]) = SA;
X_test.(['SAI_pre_' names{end}]) = SA - X_test.age;

%% Corrected AEE
% Idea 1) Subtract linear effect of age on test or val set
% Idea 2) Subtract linear effect of age on each test cohort and average
% in_c = 4;
in_m = 2;
for in_c = 1:8
    idx_data = ismember(X_test.cohort_code,in_c) & ismember(X_test.mode,in_m);
    for k = 1:length(names)
        mdl = fitlm(X_test.age(idx_data), X_test.(['SAI_' names{k}])(idx_data));
        X_test.(['SAI_c' num2str(in_c) '_' names{k}]) = X_test.(['SAI_' names{k}]) - mdl.Coefficients.Estimate(1) - mdl.Coefficients.Estimate(2)*(X_test.age);
    end
end

%% Save table
% save('C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data\X_test.mat','X_test');
% writetable(X_test, 'C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data\X_test.csv');

%% Figure test corrected aee
% in_c = 3;
% in_m = 2;
% k = 5;
% idx_data = ismember(X_test.cohort_code,in_c) & ismember(X_test.mode,in_m);
% figure
% ax1 = subplot(2,1,1);
% plot(X_test.age(idx_data), X_test.(['SAI_' names{k}])(idx_data), '.');
% hold on
% plot([20 90], [0 0], '--k');
% ax2 = subplot(2,1,2);
% plot(X_test.age(idx_data), X_test.(['SAI_c' num2str(in_c) '_' names{k}])(idx_data), '.');
% hold on
% plot([20 90], [0 0], '--k');
% linkaxes([ax1 ax2],'xy');

%% Error stats
age_ranges = 20:5:90;
in_c = 1:6;
% in_c = 8;
in_m = 2;
mean_train_age = 52.69;
mean_train_age_cohort = [37.9 41.97 58.67 81.32 44.76 68.81 mean_train_age mean_train_age];
n_age_range = nan(length(names),size(age_ranges,2)-1);
error_range = nan(length(names),5,size(age_ranges,2)-1);
error_range_baseline = [prctile(X_test.age - mean_train_age, 5),...
    mean(X_test.age - mean_train_age),...
    prctile(X_test.age - mean_train_age, 95),...
    mean(abs(X_test.age - mean_train_age)),...
    std(abs(X_test.age - mean_train_age))];
error_range_pre = nan(length(names),5,size(age_ranges,2)-1);
n_age_range_cohort = nan(length(names),size(age_ranges,2)-1,max(X_test.cohort_code));
error_range_cohort = nan(length(names),5,size(age_ranges,2)-1,max(X_test.cohort_code));
error_range_cohort_baseline = arrayfun(@(x) [prctile(X_test.age(X_test.cohort_code == x) - mean_train_age_cohort(x), 5),...
    mean(X_test.age(X_test.cohort_code == x) - mean_train_age_cohort(x)),...
    prctile(X_test.age(X_test.cohort_code == x) - mean_train_age_cohort(x), 95),...
    mean(abs(X_test.age(X_test.cohort_code == x) - mean_train_age_cohort(x))),...
    std(abs(X_test.age(X_test.cohort_code == x) - mean_train_age_cohort(x)))], 1:8, 'Un', 0);
error_range_pre_cohort = nan(length(names),5,size(age_ranges,2)-1,max(X_test.cohort_code));

for k = 1:length(names)
    for i = 1:size(age_ranges,2)-1
        if i == size(age_ranges,2)-1
            idx_age_range = X_test.age >= age_ranges(i) & X_test.age <= age_ranges(i+1) & ismember(X_test.cohort_code,in_c) & X_test.mode == in_m;
        else
            idx_age_range = X_test.age >= age_ranges(i) & X_test.age < age_ranges(i+1) & ismember(X_test.cohort_code, in_c) & X_test.mode == in_m;
        end
        % full
        n_age_range(k,i) = sum(idx_age_range);
        error_range(k,1,i) = prctile(X_test.(['SA_' names{k}])(idx_age_range), 5);
        error_range(k,2,i) = mean(X_test.(['SA_' names{k}])(idx_age_range),'omitnan');
        error_range(k,3,i) = prctile(X_test.(['SA_' names{k}])(idx_age_range), 95);
        error_range(k,4,i) = mean(abs(X_test.(['SA_' names{k}])(idx_age_range) - X_test.age(idx_age_range)),'omitnan');
        error_range(k,5,i) = std(abs(X_test.(['SA_' names{k}])(idx_age_range) - X_test.age(idx_age_range)),'omitnan');
        % pre
        error_range_pre(k,1,i) = prctile(X_test.(['SA_pre_' names{k}])(idx_age_range), 5);
        error_range_pre(k,2,i) = mean(X_test.(['SA_pre_' names{k}])(idx_age_range),'omitnan');
        error_range_pre(k,3,i) = prctile(X_test.(['SA_pre_' names{k}])(idx_age_range), 95);
        error_range_pre(k,4,i) = mean(abs(X_test.(['SA_pre_' names{k}])(idx_age_range) - X_test.age(idx_age_range)),'omitnan');
        error_range_pre(k,5,i) = std(abs(X_test.(['SA_pre_' names{k}])(idx_age_range) - X_test.age(idx_age_range)),'omitnan');
        for c = 1:max(X_test.cohort_code)
            if i == size(age_ranges,2)-1
                idx_age_range = X_test.cohort_code == c & X_test.age >= age_ranges(i) & X_test.age <= age_ranges(i+1) & X_test.mode == in_m;
            else
                idx_age_range = X_test.cohort_code == c & X_test.age >= age_ranges(i) & X_test.age < age_ranges(i+1) & X_test.mode == in_m;
            end
            % full
            n_age_range_cohort(k,i,c) = sum(idx_age_range);
            error_range_cohort(k,1,i,c) = prctile(X_test.(['SA_' names{k}])(idx_age_range), 5);
            error_range_cohort(k,2,i,c) = mean(X_test.(['SA_' names{k}])(idx_age_range),'omitnan');
            error_range_cohort(k,3,i,c) = prctile(X_test.(['SA_' names{k}])(idx_age_range), 95);
            error_range_cohort(k,4,i,c) = mean(abs(X_test.(['SA_' names{k}])(idx_age_range) - X_test.age(idx_age_range)),'omitnan');
            error_range_cohort(k,5,i,c) = std(abs(X_test.(['SA_' names{k}])(idx_age_range) - X_test.age(idx_age_range)),'omitnan');
            % pre
            error_range_pre_cohort(k,1,i,c) = prctile(X_test.(['SA_pre_' names{k}])(idx_age_range), 5);
            error_range_pre_cohort(k,2,i,c) = mean(X_test.(['SA_pre_' names{k}])(idx_age_range),'omitnan');
            error_range_pre_cohort(k,3,i,c) = prctile(X_test.(['SA_pre_' names{k}])(idx_age_range), 95);
            error_range_pre_cohort(k,4,i,c) = mean(abs(X_test.(['SA_pre_' names{k}])(idx_age_range) - X_test.age(idx_age_range)),'omitnan');
            error_range_pre_cohort(k,5,i,c) = std(abs(X_test.(['SA_pre_' names{k}])(idx_age_range) - X_test.age(idx_age_range)),'omitnan');
            
        end
    end
end

% error_range(:,:,n_age_range < 10) = nan;
% error_range_pre(:,:,n_age_range < 10) = nan;
%
% error_range_cohort(:,:,n_age_range < 10,:) = nan;
% error_range_pre_cohort(:,:,n_age_range < 10,:) = nan;
if in_c(1) == 8
    error_range8 = error_range;
    n_age_range8 = n_age_range;
else
    error_range16 = error_range;
    n_age_range16 = n_age_range;
end

%% Performance tables
perf_mae_pg = arrayfun(@(x) mean(squeeze(error_range(x,4,:)),'omitnan'), 1:size(error_range, 1));
perf_mae_pg_pre = arrayfun(@(x) mean(squeeze(error_range_pre(x,4,:)),'omitnan'), 1:size(error_range_pre, 1));
perf_mae_pg_std = arrayfun(@(x) std(squeeze(error_range(x,4,:)),'omitnan'), 1:size(error_range, 1));
perf_mae_pg_pre_std = arrayfun(@(x) std(squeeze(error_range_pre(x,4,:)),'omitnan'), 1:size(error_range_pre, 1));

perf_mae_pg_cohort = squeeze(mean(error_range_cohort(:,4,:,:),3,'omitnan'));
perf_mae_pg_pre_cohort = squeeze(mean(error_range_pre_cohort(:,4,:,:),3,'omitnan'));
perf_mae_pg_cohort_std = squeeze(std(error_range_cohort(:,4,:,:),0,3,'omitnan'));
perf_mae_pg_pre_cohort_std = squeeze(std(error_range_pre_cohort(:,4,:,:),0,3,'omitnan'));

% Show performance table
% 1) MAE: Model - Age ranges
age_ranges_str = arrayfun(@(x) ['[' num2str(x) ' - ' num2str(x + median(diff(age_ranges))) ']'], age_ranges,'Un',0);
fprintf(['MAE' repmat('\t%s',1,size(age_ranges,2)-1) '\n'],age_ranges_str{1:end-1});
fprintf(['n' repmat('\t%.0f',1,size(n_age_range(k,:),2)) '\n'],n_age_range(k,:));
for k = 1:length(names)
    fprintf('%s',names{k});
    for i = 1:size(age_ranges,2)-1
        fprintf('\t%.3g \x00B1 %.3g',error_range(k, 4, i),error_range(k, 5, i));
    end
    fprintf('\t%.3g \x00B1 %.3g\n',perf_mae_pg(k),perf_mae_pg_std(k));
end
% 1.1) MAE: only basic vertical
% for i = 1:size(age_ranges,2)-1
%     fprintf('%.3g \x00B1 %.3g\n',error_range(11, 4, i),error_range(11, 5, i));
% end
fprintf('%.3g \x00B1 %.3g\n',perf_mae_pg(11),perf_mae_pg_std(11));
% 2) MAE: Model - Cohorts
cohorts_str = unique(X_test.cohort,'Stable');
fprintf(['MAE' repmat('\t%s',1,size(cohorts_str,1)) '\n'],cohorts_str{1:end});
for k = 1:length(names)
    fprintf('%s',names{k});
    for c = 1:max(X_test.cohort_code)
        fprintf('\t%.3g',perf_mae_pg_cohort(k, c) - error_range_cohort_baseline{c}(4));
    end
    fprintf('\n');
    %     fprintf('\t%.3g \x00B1 %.3g\n',perf_mae_pg_cohort(k),perf_mae_pg_cohort_std(k));
end


%% Performance visualization for one model

k = 9;
in_c_both = {1:6, 8};
in_m = 2;

h = figure;
h.Position(3:4) = [1200 600];
centerfig(h);
for i = 1:2
    in_c = in_c_both{i};
    idx_data = ismember(X_test.cohort_code, in_c) & X_test.mode == in_m;
    if i == 1
        error_range = error_range16;
        n_age_range = n_age_range16;
    else
        error_range = error_range8;
        n_age_range = n_age_range8;
    end
    subplot(4,2,i)
    hold all
    plot(age_ranges(1:end-1) + mean(diff(age_ranges))/2, squeeze(error_range(k,4,:)),'-ok');
    plot(age_ranges(1:end-1) + mean(diff(age_ranges))/2, squeeze(error_range(k,4,:)) + (squeeze(error_range(k,5,:)) ./ sqrt((n_age_range(k,:))')),'--k');
    plot(age_ranges(1:end-1) + mean(diff(age_ranges))/2, squeeze(error_range(k,4,:)) - (squeeze(error_range(k,5,:)) ./ sqrt((n_age_range(k,:))')),'--k');
    set(gca,'XTick',age_ranges);
    ax_min = min([floor(min(X_test.age(idx_data))/5)*5 floor(min(X_test.(['SA_' names{k}])(idx_data))/5)*5]);
    ax_max = max([ceil(max(X_test.age(idx_data))/5)*5 ceil(max(X_test.(['SA_' names{k}])(idx_data))/5)*5]);
    grid on
    axis([ax_min, ax_max, 0, 20]);
    xlabel('Chronological age [years]')
    ylabel('MAE [years]')
    if i == 1
        title('a')
    else
        title('b');
    end
    
    subplot(4,2,i + (2:2:6))
    hold all
    bap = scatter(X_test.age(idx_data), X_test.(['SA_' names{k}])(idx_data),15,'k','filled');
    if sum(idx_data) > 1000
        alpha_val = 0.2;
    else
        alpha_val = 0.5;
    end
    bap.MarkerFaceAlpha = alpha_val;
    bap.MarkerEdgeAlpha = alpha_val;
    x_age = age_ranges(2:end) - median(diff(age_ranges))/2;
    plot(x_age,squeeze(error_range(k,1,:))','--m','LineWidth',2)
    plot(x_age,squeeze(error_range(k,2,:))','--m','LineWidth',2)
    plot(x_age,squeeze(error_range(k,3,:))','--m','LineWidth',2)
    plot([0 100], [0 100],'--r','LineWidth',0.25,'LineWidth',2)
    grid on
    xlabel('Chronological age [years]')
    ylabel('Age estimate [years]')
    [r, ~] = corrcoef(X_test.age(idx_data), X_test.(['SA_' names{k}])(idx_data));
    text(ax_min + 0.1*(ax_max - ax_min), ax_max - 0.1*(ax_max - ax_min), ['r = ' num2str(r(1,2))]);
%     axis equal
    axis([ax_min, ax_max, ax_min, ax_max]);
    set(gca,'XTick',age_ranges);
end
set(gcf,'Color',[1 1 1]);
set( findall(h, '-property', 'fontsize'), 'fontsize', 10);
% export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\scatter_mae_' names{k}], '-pdf', '-transparent');
export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\scatter_mae_' names{k} '_' 'both'],'-m4', '-dpng', '-transparent');
% print('-dpdf',['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\scatter_mae_' names{k} '_' num2str(in_c)]);
% print('-bestfit','-dpdf',['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\scatter_mae_' names{k} '_' 'both']);

%% Sleep summary statistics table
%  Metrics: N, age, bmi, tst, sl, waso, se, n1, n2, n3, rem, ari, ahi, plmi
M_sm_n_m = [arrayfun(@(x) sum(X_test.cohort_code == x), 1:8); ...
    arrayfun(@(x) sum(X_test.cohort_code == x & X_test.mode == 0), 1:8); ...
    arrayfun(@(x) sum(X_test.cohort_code == x & X_test.mode == 1), 1:8); ...
    arrayfun(@(x) sum(X_test.cohort_code == x & X_test.mode == 2), 1:8)];
M_sm_n_s = [arrayfun(@(x) sum(X_test.cohort_code == x) / M_sm_n_m(1,x), 1:8); ...
    arrayfun(@(x) sum(X_test.cohort_code == x & X_test.mode == 0) / M_sm_n_m(1,x), 1:8); ...
    arrayfun(@(x) sum(X_test.cohort_code == x & X_test.mode == 1) / M_sm_n_m(1,x), 1:8); ...
    arrayfun(@(x) sum(X_test.cohort_code == x & X_test.mode == 2) / M_sm_n_m(1,x), 1:8)];

metric_fields = {'age','sex','bmi','tst','sl','waso','se','n1p','n2p','n3p','remp','ari','ahi','plmi','O280'};
var_nom = [0; 1; zeros(length(metric_fields)-2,1)];
M_sm_v_m = zeros(length(metric_fields),8);
M_sm_v_s = zeros(length(metric_fields),8);

for i = 1:length(metric_fields)
    if var_nom(i) == 1
        M_sm_v_m(i,:) = arrayfun(@(x) sum(X_test.(metric_fields{i})(X_test.cohort_code == x), 'omitnan'), 1:8);
        M_sm_v_s(i,:) = arrayfun(@(x) mean(X_test.(metric_fields{i})(X_test.cohort_code == x), 'omitnan'), 1:8);
    else
        M_sm_v_m(i,:) = arrayfun(@(x) mean(X_test.(metric_fields{i})(X_test.cohort_code == x), 'omitnan'), 1:8);
        M_sm_v_s(i,:) = arrayfun(@(x) std(X_test.(metric_fields{i})(X_test.cohort_code == x), 'omitnan'), 1:8);
    end
end

T_sm = array2table([[M_sm_n_m, M_sm_n_s]; [M_sm_v_m, M_sm_v_s]]);
T_sm.Properties.RowNames = [{'N','N_test','N_val','N_train'}, metric_fields];
var_nom = [ones(4,1); var_nom];

cohorts_str = unique(X_test.cohort,'Stable');
fprintf(['\t' repmat('\t%s',1,size(cohorts_str,1)) '\n'],cohorts_str{1:end});
for i = 1:size(T_sm,1)
    fprintf('%s\t',T_sm.Properties.RowNames{i});
    if var_nom(i) == 1
        fprintf('n, (%%)\t');
    else
        fprintf('\x03bc \x00B1 \x03c3\t');
    end
    for g = 1:length(cohorts_str)
        if var_nom(i) == 1
            if 100*T_sm{i,2*g-1} < 1
                format = '%.3f';
            elseif 100*T_sm{i,2*g-1} < 10
                format = '%.2f';
            else
                format = '%.1f';
            end
            fprintf(['%.0f, (' format ' %%)\t'],T_sm{i,g},100*T_sm{i,g+length(cohorts_str)})
        else
            if T_sm{i,2*g-1} < 1
                format1 = '%.3f';
            elseif T_sm{i,2*g-1} < 10
                format1 = '%.2f';
            else
                format1 = '%.1f';
            end
            if T_sm{i,2*g} < 1
                format2 = '%.3f';
            elseif T_sm{i,2*g} < 10
                format2 = '%.2f';
            else
                format2 = '%.1f';
            end
            fprintf([format1 ' \x00B1 ' format2 '\t'],T_sm{i,g},T_sm{i,g+length(cohorts_str)})
        end
    end
    fprintf('\n');
end

%% Mortality analysis - Cox's hazard + kaplein meyer (SHHS)

% Race
X_test.race_01 = double(X_test.race == 1);
X_test.race_01(isnan(X_test.race)) = nan;
% Smoking
X_test.smoke_02 = double(X_test.smoke == 2);
X_test.smoke_02(isnan(X_test.smoke)) = nan;
X_test.smoke_01 = double(X_test.smoke == 1);
X_test.smoke_01(isnan(X_test.smoke)) = nan;
% Married
X_test.marital_01 = double(X_test.marital == 1);
X_test.marital_01(isnan(X_test.marital)) = nan;
% Education
X_test.edu_01 = double(X_test.edu == 1);
X_test.edu_01(isnan(X_test.edu)) = nan;
X_test.edu_02 = double(X_test.edu == 2);
X_test.edu_02(isnan(X_test.edu)) = nan;
X_test.edu_03 = double(X_test.edu == 3);
X_test.edu_03(isnan(X_test.edu)) = nan;
X_test.edu_04 = double(X_test.edu == 4);
X_test.edu_04(isnan(X_test.edu)) = nan;

% Time to death adjusted for age
X_test.t_death_adj = X_test.t_death + 365*(X_test.age - min(X_test.age(X_test.cohort_code == 6)));

% Collect HR
HR_SHHS = cell(length(names),3);

% To impute?
to_impute = 1;

% Verbose imputation stats
verbose_impute = 1;

% Mortality type:
mortality_type = 'vital';

% Select Model
% k = 3;
for covars = 0:2
    for k = 1:length(names)
        var_r = {['SAI_' names{k}]};
        %         var_r = {['SAI_c' num2str(6) '_' names{k}]};
        % covars = 0;
        
        % Response variable
        idx_rv = cellfun(@(x) find(ismember(X_test.Properties.VariableNames, x)), {'t_death'});
        
        % Predictors
        idx_pv_aee = find(ismember(X_test.Properties.VariableNames,{var_r{1}}));
        idx_pv_0 = find(ismember(X_test.Properties.VariableNames,{'age'}));
        idx_pv_1 = find(ismember(X_test.Properties.VariableNames,{'age','sex','bmi','race_01','smoke_01','smoke_02','edu_02','edu_03','edu_04','alch','caf','ben','ant'}));
        idx_pv_2 = find(ismember(X_test.Properties.VariableNames,{'remp','ari','ahi','O280','waso','n2p','ESS','hype','chf','copd','t2d','haa','stro'}));
        if covars == 1
            idx_pv = idx_pv_1;
        elseif covars == 2
            idx_pv = [idx_pv_1 idx_pv_2];
        elseif covars == 0
            idx_pv = idx_pv_0;
        end
        
        % Select data subset
        idx_data = X_test.cohort_code == 6 & X_test.v1 == 1;
        
        % Data imputation
        x_data = X_test{:, [idx_rv idx_pv_aee idx_pv]};
        x_data_missing = X_test{idx_data, idx_pv};
        x_data_imp = x_data;
        if to_impute == 1 && covars > 0
            if k == 1
                if verbose_impute == 1
                    x_perc_1 = prctile(x_data_missing,[5, 50, 95]);
                    x_mean_1 = mean(x_data_missing,'omitnan');
                end
                writematrix(x_data_missing, "C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data\data_missing.csv");
                system('"C:\Program Files\R\R-4.0.4\bin\R.exe" CMD BATCH impute_data.R');
            end
            x_data_imputed = readmatrix("C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data\data_imputed.csv");
            if verbose_impute == 1
                x_perc_2 = prctile(x_data_imputed(:,2:end),[5, 50, 95]);
                x_mean_2 = mean(x_data_imputed(:,2:end),'omitnan');
                x_change = [(x_mean_2 - x_mean_1)./(x_mean_1 + eps); (x_perc_2 - x_perc_1)./(x_perc_1 + eps)];
            end
            x_data_imp(idx_data, 3:end) = x_data_imputed(:,2:end);
        end
        idx_pv_all = [idx_pv_aee idx_pv];
        
        % Select data subset
        idx_data = X_test.cohort_code == 6 & ~any(isnan(x_data_imp),2) & ~isnan(X_test.(mortality_type)) & X_test.v1 == 1;
        idx_data_wbl = X_test.cohort_code == 6 & ~any(isnan(x_data_imp),2) & (x_data_imp(:,1) ~= 0) & ~isnan(X_test.(mortality_type)) & X_test.v1 == 1;
        
        % Variable type summary
        pv_std = std(x_data_imp(idx_data, 2:end));
        idx_binary = arrayfun(@(x) ~any(unique(x_data_imp(idx_data,x)) ~= 0 & unique(x_data_imp(idx_data,x)) ~= 1), 2:size(x_data_imp,2));
        pv_std(idx_binary) = 1;
        
        % Proportional Hazards model
        [b,logl,H,stats] = coxphfit(x_data_imp(idx_data,2:end),x_data_imp(idx_data,1),'Censoring',X_test.(mortality_type)(idx_data));
        [parmHat, parmCI] = wblfit(x_data_imp(idx_data_wbl,1), 0.05, X_test.(mortality_type)(idx_data_wbl));
        fprintf(['Cox proportional hazards regression. %s ~ %s', repmat(' + %s',1,length(idx_pv_all)-1), '.\n'],X_test.Properties.VariableNames{[idx_rv idx_pv_all]});
        fprintf('\t\tHR (CI)\tB\tp\n');
        for i = 1:length(idx_pv_all)
            CI_raw = (stats.beta(i) + [-1 1]*1.96*stats.se(i));
            CI = exp(pv_std(i) * CI_raw);
            fprintf('%s\t\t%.2f, (%.2f, %.2f)\t%.5f\t%.5g\n',X_test.Properties.VariableNames{idx_pv_all(i)},exp(pv_std(i)*stats.beta(i)),CI(1),CI(2),stats.beta(i),stats.p(i));
        end
        
        SAI_var_idx = contains(X_test.Properties.VariableNames(idx_pv_all), var_r{1});
        %         SAI_var_std = pv_std(SAI_var_idx);
        SAI_var_std = 10;
        SAI_var_beta = [stats.beta(SAI_var_idx) stats.beta(SAI_var_idx) + [-1 1]*1.96*stats.se(SAI_var_idx)];
        
        h = figure;
        h.Position(3:4) = [600 400];
        centerfig(h);
        hold all
        plot_cell = cell(3,1);
        plot_cell{1} = stairs(H(:,1)/365,exp(-H(:,2) * exp(-SAI_var_std * SAI_var_beta(1))),'b','LineWidth',2);
        stairs(H(:,1)/365,exp(-H(:,2) * exp(-SAI_var_std * SAI_var_beta(2))),'--b','LineWidth',1);
        stairs(H(:,1)/365,exp(-H(:,2) * exp(-SAI_var_std * SAI_var_beta(3))),'--b','LineWidth',1);
        plot_cell{2} = stairs(H(:,1)/365,exp(-H(:,2) * exp(SAI_var_std * SAI_var_beta(1))),'r','LineWidth',2);
        stairs(H(:,1)/365,exp(-H(:,2) * exp(SAI_var_std * SAI_var_beta(2))),'--r','LineWidth',1);
        stairs(H(:,1)/365,exp(-H(:,2) * exp(SAI_var_std * SAI_var_beta(3))),'--r','LineWidth',1);
        plot_cell{3} = plot(0,1,'--k');
        grid minor
        xlabel('Follow-up [years]')
        ylabel('Survival %')
        %         legend(horzcat(plot_cell{:}),{'-1 \sigma(AEE)','+1 \sigma(AEE)','95 % CI'},'Location','northeast')
        legend(horzcat(plot_cell{:}),{'-10 AEE','+10 AEE','95 % CI'},'Location','northeast')
        set(gcf,'Color',[1 1 1]);
        set( findall(h, '-property', 'fontsize'), 'fontsize', 10);
        export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\survival_curve_shhs_' names{k} '_model_' num2str(covars) '_mt_' mortality_type], '-pdf', '-transparent');
        
        HR_SHHS{k,covars + 1} = exp(SAI_var_beta*SAI_var_std);
    end
end

for k = 1:size(HR_SHHS,1)
    fprintf('%s\t%.2f, (%.2f, %.2f)\t%.2f, (%.2f, %.2f)\t%.2f, (%.2f, %.2f)\n',names{k},HR_SHHS{k,:});
end

%% Mortality analysis - Summary table (SHHS)
in_c = 6;
in_m = 0:2;
k = 9;
var_r = {['SAI_c' num2str(in_c) '_' names{k}]};
idx_data = ismember(X_test.cohort_code, in_c) & ismember(X_test.mode, in_m) & X_test.v1 == 1;
AEEc = X_test.(var_r{1})(idx_data);
AEEc_Q = prctile(AEEc, [0 25, 50, 75 100]);
AEEc_Q(end) = AEEc_Q(end) + 0.01;
N_Q = arrayfun(@(x) sum(AEEc >= AEEc_Q(x) & AEEc < AEEc_Q(x+1)), 1:4);
var_nom = arrayfun(@(x) ~any(unique(X_test{idx_data & ~isnan(X_test{:,x}),x}) ~= 0 & unique(X_test{idx_data & ~isnan(X_test{:,x}),x}) ~= 1), idx_pv);

fprintf(['\t' repmat('\tQ%.0f',1,4) '\n'],1:4);
fprintf(['\t' repmat('\t%.1f < AEEc <= %.1f',1,4) '\n'],[AEEc_Q(1), repelem(AEEc_Q(2:end-1),2), AEEc_Q(end)]);
fprintf(['\t' repmat('\t(n = %.0f)',1,4) '\n'],N_Q);
for i = 1:size(idx_pv,2)
    fprintf('%s\t', X_test.Properties.VariableNames{idx_pv(i)});
    if var_nom(i) == 1
        fprintf('n, (%%)\t');
    else
        fprintf('\x03bc \x00B1 \x03c3\t');
    end
    var_data = X_test{idx_data, idx_pv(i)};
    for g = 1:4
        var_data_Q = var_data(AEEc > AEEc_Q(g) & AEEc <= AEEc_Q(g+1));
        if var_nom(i) == 1
            if 100*mean(var_data_Q,'omitnan') < 1
                format = '%.3f';
            elseif 100*mean(var_data_Q,'omitnan') < 10
                format = '%.2f';
            else
                format = '%.1f';
            end
            fprintf(['%.0f, (' format ' %%)\t'], sum(var_data_Q,'omitnan'), 100*mean(var_data_Q,'omitnan'))
        else
            if mean(var_data_Q,'omitnan') < 1
                format1 = '%.3f';
            elseif mean(var_data_Q,'omitnan') < 10
                format1 = '%.2f';
            else
                format1 = '%.1f';
            end
            if std(var_data_Q,'omitnan') < 1
                format2 = '%.3f';
            elseif std(var_data_Q,'omitnan') < 10
                format2 = '%.2f';
            else
                format2 = '%.1f';
            end
            fprintf([format1 ' \x00B1 ' format2 '\t'],mean(var_data_Q,'omitnan'),std(var_data_Q,'omitnan'))
        end
    end
    fprintf('\n');
end

%% Mortality analysis - Cox's hazard + kaplein meyer (WSC)
% train_data = readtable("H:\nAge\X_train.csv");
% wsc_train_subjects = cellfun(@(x) x(1:end-2), train_data.names(train_data.cohort_code == 3),'Un',0);

% Race
X_test.race_01 = double(X_test.race == 1);
X_test.race_01(isnan(X_test.race)) = nan;
% Smoking
X_test.smoke_02 = double(X_test.smoke == 2);
X_test.smoke_02(isnan(X_test.smoke)) = nan;
X_test.smoke_01 = double(X_test.smoke == 1);
X_test.smoke_01(isnan(X_test.smoke)) = nan;

% Time to death adjusted for age
X_test.t_death_adj = X_test.t_death + 365*(X_test.age - min(X_test.age(X_test.cohort_code == 3)));

to_impute = 1;

% Mortality type:
mortality_type = 'vitalC';

% Collect HR
HR_WSC = cell(length(names),3);

% Select Model
% k = 3;
for covars = 0:2
    for k = 1:length(names)
        % k = 10;
        var_r = {['SAI_' names{k}]};
        %         var_r = {['SAI_c' num2str(3) '_' names{k}]};
        % covars = 2;
        
        % Response variable
        idx_rv = cellfun(@(x) find(ismember(X_test.Properties.VariableNames, x)), {'t_death'});
        
        % Predictors
        idx_pv_aee = find(ismember(X_test.Properties.VariableNames,{var_r{1}}));
        idx_pv_0 = find(ismember(X_test.Properties.VariableNames,{'age'}));
        idx_pv_1 = find(ismember(X_test.Properties.VariableNames,{'age','sex','bmi','race_01','smoke_01','smoke_02','edu_02','edu_03','edu_04','alch','caf','ben','ant','sme'}));
        idx_pv_2 = find(ismember(X_test.Properties.VariableNames,{'remp','ari','ahi','O280','waso','n2p','ESS','hype','chf','t2d','haa','stro'}));
        if covars == 1
            idx_pv = idx_pv_1;
        elseif covars == 2
            idx_pv = [idx_pv_1 idx_pv_2];
        elseif covars == 0
            idx_pv = idx_pv_0;
        end
        
        % Exclude subjects in training data
        %         idx_in_train = cellfun(@(x) any(strcmp(wsc_train_subjects, x(1:end-2))), X_test.names);
        
        % Select data subset (TODO: Choose first or latest sample?)
%         idx_first_visit_all = zeros(size(X_test.age));
%         unique_subj_id = unique(cellfun(@(x) x(1:end-2), X_test.names(X_test.cohort_code == 3),'Un',0));
%         subj_id_short = cellfun(@(x) x(1:end-2), X_test.names, 'Un', 0);
%         for i = 1:length(unique_subj_id)
%             idx_id = find(strcmp(subj_id_short, unique_subj_id{i}));
%             idx_id = idx_id(X_test.age(idx_id) == min(X_test.age(idx_id)));
%             idx_first_visit_all(idx_id) = 1;
%         end
        idx_data = X_test.cohort_code == 3 & X_test.v1 == 1; % & ~idx_in_train;
        
        % Data imputation
        x_data = X_test{:, [idx_rv idx_pv_aee idx_pv]};
        x_data_missing = X_test{idx_data, idx_pv};
        x_data_imp = x_data;
        if to_impute == 1 && covars > 0
            if k == 1
                writematrix(x_data_missing, "C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data\data_missing.csv");
                system('"C:\Program Files\R\R-4.0.4\bin\R.exe" CMD BATCH impute_data.R');
            end
            x_data_imputed = readmatrix("C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data\data_imputed.csv");
            x_data_imp(idx_data, 3:end) = x_data_imputed(:,2:end);
        end
        idx_pv_all = [idx_pv_aee idx_pv];
        
        % Select data subset
        idx_data = X_test.cohort_code == 3 & ~any(isnan(x_data_imp),2) & ~isnan(X_test.(mortality_type)) & X_test.v1 == 1;% & ~idx_in_train;
        idx_data_wbl = X_test.cohort_code == 3 & ~any(isnan(x_data_imp),2) & (x_data_imp(:,1) ~= 0) & ~isnan(X_test.(mortality_type)) & X_test.v1 == 1;% & ~idx_in_train;
        
        % Variable type summary
        pv_std = std(x_data_imp(idx_data, 2:end));
        idx_binary = arrayfun(@(x) ~any(unique(x_data_imp(idx_data,x)) ~= 0 & unique(x_data_imp(idx_data,x)) ~= 1), 2:size(x_data_imp,2));
        pv_std(idx_binary) = 1;
        
        % Proportional Hazards model
        [b,logl,H,stats] = coxphfit(x_data_imp(idx_data,2:end),x_data_imp(idx_data,1),'Censoring',X_test.(mortality_type)(idx_data));
        [parmHat, parmCI] = wblfit(x_data_imp(idx_data_wbl,1), 0.05, X_test.(mortality_type)(idx_data_wbl));
        fprintf(['Cox proportional hazards regression. %s ~ %s', repmat(' + %s',1,length(idx_pv_all)-1), '.\n'],X_test.Properties.VariableNames{[idx_rv idx_pv_all]});
        fprintf('\t\tHR (CI)\tB\tp\n');
        for i = 1:length(idx_pv_all)
            CI_raw = (stats.beta(i) + [-1 1]*1.96*stats.se(i));
            CI = exp(pv_std(i) * CI_raw);
            fprintf('%s\t\t%.2f, (%.2f, %.2f)\t%.5f\t%.5g\n',X_test.Properties.VariableNames{idx_pv_all(i)},exp(pv_std(i)*stats.beta(i)),CI(1),CI(2),stats.beta(i),stats.p(i));
        end
        
        SAI_var_idx = contains(X_test.Properties.VariableNames(idx_pv_all), var_r{1});
        %         SAI_var_std = pv_std(SAI_var_idx);
        SAI_var_std = 10;
        SAI_var_beta = [stats.beta(SAI_var_idx) stats.beta(SAI_var_idx) + [-1 1]*1.96*stats.se(SAI_var_idx)];
        
        h = figure;
        h.Position(3:4) = [600 400];
        centerfig(h);
        hold all
        plot_cell = cell(3,1);
        plot_cell{1} = stairs(H(:,1)/365,exp(-H(:,2) * exp(-SAI_var_std * SAI_var_beta(1))),'b','LineWidth',2);
        stairs(H(:,1)/365,exp(-H(:,2) * exp(-SAI_var_std * SAI_var_beta(2))),'--b','LineWidth',1);
        stairs(H(:,1)/365,exp(-H(:,2) * exp(-SAI_var_std * SAI_var_beta(3))),'--b','LineWidth',1);
        plot_cell{2} = stairs(H(:,1)/365,exp(-H(:,2) * exp(SAI_var_std * SAI_var_beta(1))),'r','LineWidth',2);
        stairs(H(:,1)/365,exp(-H(:,2) * exp(SAI_var_std * SAI_var_beta(2))),'--r','LineWidth',1);
        stairs(H(:,1)/365,exp(-H(:,2) * exp(SAI_var_std * SAI_var_beta(3))),'--r','LineWidth',1);
        plot_cell{3} = plot(0,1,'--k');
        grid minor
        xlabel('Follow-up [years]')
        ylabel('Survival Probability')
        %         legend(horzcat(plot_cell{:}),{'-1 \sigma(AEE)','+1 \sigma(AEE)','95 % CI'},'Location','northeast')
        legend(horzcat(plot_cell{:}),{'-10 AEE','+10 AEE','95 % CI'},'Location','northeast')
        set(gcf,'Color',[1 1 1]);
        set( findall(h, '-property', 'fontsize'), 'fontsize', 10);
        export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\survival_curve_wsc_' names{k} '_model_' num2str(covars) '_mt_' mortality_type], '-pdf', '-transparent');
        HR_WSC{k,covars + 1} = exp(SAI_var_beta*SAI_var_std);
    end
end

for k = 1:size(HR_WSC,1)
    fprintf('%s\t%.2f, (%.2f, %.2f)\t%.2f, (%.2f, %.2f)\t%.2f, (%.2f, %.2f)\n',names{k},HR_WSC{k,:});
end

%% Mortality - Summary table (WSC)
in_c = 3;
in_m = 0:2;
k = 9;
var_r = {['SAI_c' num2str(in_c) '_' names{k}]};
idx_data = ismember(X_test.cohort_code, in_c) & ismember(X_test.mode, in_m) & X_test.v1 == 1;
AEEc = X_test.(var_r{1})(idx_data);
AEEc_Q = prctile(AEEc, [0 25, 50, 75 100]);
AEEc_Q(end) = AEEc_Q(end) + 0.01;
N_Q = arrayfun(@(x) sum(AEEc >= AEEc_Q(x) & AEEc < AEEc_Q(x+1)), 1:4);
var_nom = arrayfun(@(x) ~any(unique(X_test{idx_data & ~isnan(X_test{:,x}),x}) ~= 0 & unique(X_test{idx_data & ~isnan(X_test{:,x}),x}) ~= 1), idx_pv);

fprintf(['\t' repmat('\tQ%.0f',1,4) '\n'],1:4);
fprintf(['\t' repmat('\t%.1f < AEEc <= %.1f',1,4) '\n'],[AEEc_Q(1), repelem(AEEc_Q(2:end-1),2), AEEc_Q(end)]);
fprintf(['\t' repmat('\t(n = %.0f)',1,4) '\n'],N_Q);
for i = 1:size(idx_pv,2)
    fprintf('%s\t', X_test.Properties.VariableNames{idx_pv(i)});
    if var_nom(i) == 1
        fprintf('n, (%%)\t');
    else
        fprintf('\x03bc \x00B1 \x03c3\t');
    end
    var_data = X_test{idx_data, idx_pv(i)};
    for g = 1:4
        var_data_Q = var_data(AEEc > AEEc_Q(g) & AEEc <= AEEc_Q(g+1));
        if var_nom(i) == 1
            if 100*mean(var_data_Q,'omitnan') < 1
                format = '%.3f';
            elseif 100*mean(var_data_Q,'omitnan') < 10
                format = '%.2f';
            else
                format = '%.1f';
            end
            fprintf(['%.0f, (' format ' %%)\t'], sum(var_data_Q,'omitnan'), 100*mean(var_data_Q,'omitnan'))
        else
            if mean(var_data_Q,'omitnan') < 1
                format1 = '%.3f';
            elseif mean(var_data_Q,'omitnan') < 10
                format1 = '%.2f';
            else
                format1 = '%.1f';
            end
            if std(var_data_Q,'omitnan') < 1
                format2 = '%.3f';
            elseif std(var_data_Q,'omitnan') < 10
                format2 = '%.2f';
            else
                format2 = '%.1f';
            end
            fprintf([format1 ' \x00B1 ' format2 '\t'],mean(var_data_Q,'omitnan'),std(var_data_Q,'omitnan'))
        end
    end
    fprintf('\n');
end

%% Mortality analysis - Cox's hazard + kaplein meyer (MrOS)

% Time to death adjusted for age
X_test.t_death_adj = X_test.t_death + 365*(X_test.age - min(X_test.age(X_test.cohort_code == 6)));

% Collect HR
HR_MrOS = cell(length(names),3);

% Mortality type:
mortality_type = 'vitalC';

% To impute?
to_impute = 1;

% Verbose imputation stats
verbose_impute = 1;

% Select Model
% k = 3;
for covars = 0:2
    for k = 1:length(names)
        var_r = {['SAI_' names{k}]};
        %         var_r = {['SAI_c' num2str(6) '_' names{k}]};
        % covars = 0;
        
        % Response variable
        idx_rv = cellfun(@(x) find(ismember(X_test.Properties.VariableNames, x)), {'t_death'});
        
        % Predictors
        idx_pv_aee = find(ismember(X_test.Properties.VariableNames,{var_r{1}}));
        idx_pv_0 = find(ismember(X_test.Properties.VariableNames,{'age'}));
        idx_pv_1 = find(ismember(X_test.Properties.VariableNames,{'age','bmi','race_01','smoke_01','smoke_02','edu_02','edu_03','edu_04','alch','caf','ben','ant'}));
        idx_pv_2 = find(ismember(X_test.Properties.VariableNames,{'remp','ari','ahi','O280','waso','n2p','ESS','hype','chf','copd','t2d','haa','stro','ams','awaso','mmse','pase'}));
        if covars == 1
            idx_pv = idx_pv_1;
        elseif covars == 2
            idx_pv = [idx_pv_1 idx_pv_2];
        elseif covars == 0
            idx_pv = idx_pv_0;
        end
        
        % Select data subset
        idx_data = X_test.cohort_code == 4;
        
        % Data imputation
        x_data = X_test{:, [idx_rv idx_pv_aee idx_pv]};
        x_data_missing = X_test{idx_data, idx_pv};
        x_data_imp = x_data;
        if to_impute == 1 && covars > 0
            if k == 1
                if verbose_impute == 1
                    x_perc_1 = prctile(x_data_missing,[5, 50, 95]);
                    x_mean_1 = mean(x_data_missing,'omitnan');
                end
                writematrix(x_data_missing, "C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data\data_missing.csv");
                system('"C:\Program Files\R\R-4.0.4\bin\R.exe" CMD BATCH impute_data.R');
            end
            x_data_imputed = readmatrix("C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data\data_imputed.csv");
            if verbose_impute == 1
                x_perc_2 = prctile(x_data_imputed(:,2:end),[5, 50, 95]);
                x_mean_2 = mean(x_data_imputed(:,2:end),'omitnan');
                x_change = [(x_mean_2 - x_mean_1)./(x_mean_1 + eps); (x_perc_2 - x_perc_1)./(x_perc_1 + eps)];
            end
            x_data_imp(idx_data, 3:end) = x_data_imputed(:,2:end);
        end
        idx_pv_all = [idx_pv_aee idx_pv];
        
        % Select data subset
        idx_data = X_test.cohort_code == 4 & ~any(isnan(x_data_imp),2) & ~isnan(X_test.(mortality_type));
        idx_data_wbl = X_test.cohort_code == 4 & ~any(isnan(x_data_imp),2) & (x_data_imp(:,1) ~= 0) & ~isnan(X_test.(mortality_type));
        
        % Variable type summary
        pv_std = std(x_data_imp(idx_data, 2:end));
        idx_binary = arrayfun(@(x) ~any(unique(x_data_imp(idx_data,x)) ~= 0 & unique(x_data_imp(idx_data,x)) ~= 1), 2:size(x_data_imp,2));
        pv_std(idx_binary) = 1;
        
        % Proportional Hazards model
        [b,logl,H,stats] = coxphfit(x_data_imp(idx_data,2:end),x_data_imp(idx_data,1),'Censoring',X_test.(mortality_type)(idx_data));
        [parmHat, parmCI] = wblfit(x_data_imp(idx_data_wbl,1), 0.05, X_test.(mortality_type)(idx_data_wbl));
        fprintf(['Cox proportional hazards regression. %s ~ %s', repmat(' + %s',1,length(idx_pv_all)-1), '.\n'],X_test.Properties.VariableNames{[idx_rv idx_pv_all]});
        fprintf('\t\tHR (CI)\tB\tp\n');
        for i = 1:length(idx_pv_all)
            CI_raw = (stats.beta(i) + [-1 1]*1.96*stats.se(i));
            CI = exp(pv_std(i) * CI_raw);
            fprintf('%s\t\t%.2f, (%.2f, %.2f)\t%.5f\t%.5g\n',X_test.Properties.VariableNames{idx_pv_all(i)},exp(pv_std(i)*stats.beta(i)),CI(1),CI(2),stats.beta(i),stats.p(i));
        end
        
        SAI_var_idx = contains(X_test.Properties.VariableNames(idx_pv_all), var_r{1});
        %         SAI_var_std = pv_std(SAI_var_idx);
        SAI_var_std = 10;
        SAI_var_beta = [stats.beta(SAI_var_idx) stats.beta(SAI_var_idx) + [-1 1]*1.96*stats.se(SAI_var_idx)];
        
        h = figure;
        h.Position(3:4) = [600 400];
        centerfig(h);
        hold all
        plot_cell = cell(3,1);
        plot_cell{1} = stairs(H(:,1)/365,exp(-H(:,2) * exp(-SAI_var_std * SAI_var_beta(1))),'b','LineWidth',2);
        stairs(H(:,1)/365,exp(-H(:,2) * exp(-SAI_var_std * SAI_var_beta(2))),'--b','LineWidth',1);
        stairs(H(:,1)/365,exp(-H(:,2) * exp(-SAI_var_std * SAI_var_beta(3))),'--b','LineWidth',1);
        plot_cell{2} = stairs(H(:,1)/365,exp(-H(:,2) * exp(SAI_var_std * SAI_var_beta(1))),'r','LineWidth',2);
        stairs(H(:,1)/365,exp(-H(:,2) * exp(SAI_var_std * SAI_var_beta(2))),'--r','LineWidth',1);
        stairs(H(:,1)/365,exp(-H(:,2) * exp(SAI_var_std * SAI_var_beta(3))),'--r','LineWidth',1);
        plot_cell{3} = plot(0,1,'--k');
        grid minor
        xlabel('Follow-up [years]')
        ylabel('Survival %')
        %         legend(horzcat(plot_cell{:}),{'-1 \sigma(AEE)','+1 \sigma(AEE)','95 % CI'},'Location','northeast')
        legend(horzcat(plot_cell{:}),{'-10 AEE','+10 AEE','95 % CI'},'Location','northeast')
        set(gcf,'Color',[1 1 1]);
        set( findall(h, '-property', 'fontsize'), 'fontsize', 10);
        export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\survival_curve_mros_' names{k} '_model_' num2str(covars)], '-pdf', '-transparent');
        
        HR_MrOS{k,covars + 1} = exp(SAI_var_beta*SAI_var_std);
    end
end

for k = 1:size(HR_MrOS,1)
    fprintf('%s\t%.2f, (%.2f, %.2f)\t%.2f, (%.2f, %.2f)\t%.2f, (%.2f, %.2f)\n',names{k},HR_MrOS{k,:});
end

%% Mortality analysis - Summary table (MrOS)
in_c = 4;
in_m = 0:2;
k = 9;
var_r = {['SAI_c' num2str(in_c) '_' names{k}]};
idx_data = ismember(X_test.cohort_code, in_c) & ismember(X_test.mode, in_m);
AEEc = X_test.(var_r{1})(idx_data);
AEEc_Q = prctile(AEEc, [0 25, 50, 75 100]);
AEEc_Q(end) = AEEc_Q(end) + 0.01;
N_Q = arrayfun(@(x) sum(AEEc >= AEEc_Q(x) & AEEc < AEEc_Q(x+1)), 1:4);
var_nom = arrayfun(@(x) ~any(unique(X_test{idx_data & ~isnan(X_test{:,x}),x}) ~= 0 & unique(X_test{idx_data & ~isnan(X_test{:,x}),x}) ~= 1), idx_pv);

fprintf(['\t' repmat('\tQ%.0f',1,4) '\n'],1:4);
fprintf(['\t' repmat('\t%.1f < AEEc <= %.1f',1,4) '\n'],[AEEc_Q(1), repelem(AEEc_Q(2:end-1),2), AEEc_Q(end)]);
fprintf(['\t' repmat('\t(n = %.0f)',1,4) '\n'],N_Q);
for i = 1:size(idx_pv,2)
    fprintf('%s\t', X_test.Properties.VariableNames{idx_pv(i)});
    if var_nom(i) == 1
        fprintf('n, (%%)\t');
    else
        fprintf('\x03bc \x00B1 \x03c3\t');
    end
    var_data = X_test{idx_data, idx_pv(i)};
    for g = 1:4
        var_data_Q = var_data(AEEc > AEEc_Q(g) & AEEc <= AEEc_Q(g+1));
        if var_nom(i) == 1
            if 100*mean(var_data_Q,'omitnan') < 1
                format = '%.3f';
            elseif 100*mean(var_data_Q,'omitnan') < 10
                format = '%.2f';
            else
                format = '%.1f';
            end
            fprintf(['%.0f, (' format ' %%)\t'], sum(var_data_Q,'omitnan'), 100*mean(var_data_Q,'omitnan'))
        else
            if mean(var_data_Q,'omitnan') < 1
                format1 = '%.3f';
            elseif mean(var_data_Q,'omitnan') < 10
                format1 = '%.2f';
            else
                format1 = '%.1f';
            end
            if std(var_data_Q,'omitnan') < 1
                format2 = '%.3f';
            elseif std(var_data_Q,'omitnan') < 10
                format2 = '%.2f';
            else
                format2 = '%.1f';
            end
            fprintf([format1 ' \x00B1 ' format2 '\t'],mean(var_data_Q,'omitnan'),std(var_data_Q,'omitnan'))
        end
    end
    fprintf('\n');
end

%% Mortality analysis - Cox's hazard + kaplein meyer (ALL)
% Cohort
X_test.wsc_cohort = double(X_test.cohort_code == 3);
X_test.wsc_cohort(isnan(X_test.cohort_code)) = nan;
X_test.shhs_cohort = double(X_test.cohort_code == 6);
X_test.shhs_cohort(isnan(X_test.cohort_code)) = nan;
X_test.mros_cohort = double(X_test.cohort_code == 4);
X_test.mros_cohort(isnan(X_test.cohort_code)) = nan;

% Cohorts
in_c = [3 4 6];

% Collect HR
HR_ALL = cell(length(names),3);

% Collect LE
LE_AGE = 10:10:100;
LE_ALL = cell(length(names),3, length(LE_AGE));

% Mortality type:
mortality_type = 'vitalC';

% No hypertension?
use_hyp = true;

% No sleep apnea?
use_ahi = false;

% To impute?
to_impute = 1;

% Verbose imputation stats
verbose_impute = 0;

% Select Model
% k = 3;
for covars = 0:2
    for k = 1:length(names)
        var_r = {['SAI_' names{k}]};
        %         var_r = {['SAI_c' num2str(6) '_' names{k}]};
        % covars = 0;
        
        % Response variable
        idx_rv = cellfun(@(x) find(ismember(X_test.Properties.VariableNames, x)), {'t_death'});
        
        % Predictors
        idx_pv_aee = find(ismember(X_test.Properties.VariableNames,{var_r{1}}));
        idx_pv_0 = find(ismember(X_test.Properties.VariableNames,{'age'}));
        idx_pv_1 = find(ismember(X_test.Properties.VariableNames,{'age','sex','bmi','race_01','smoke_01','smoke_02','alch','caf','ben','ant','wsc_cohort','shhs_cohort'}));
        idx_pv_2 = find(ismember(X_test.Properties.VariableNames,{'remp','ari','ahi','O280','waso','n2p','ESS','t2d','haa','stro','chf'}));
        if use_hyp
            idx_pv_2 = [idx_pv_2 find(ismember(X_test.Properties.VariableNames,{'hype'}))];
            idx_data_hyp = true(size(X_test,1),1);
        else
            idx_data_hyp = X_test.hype == 0;
        end
        if use_ahi
            idx_data_ahi = true(size(X_test,1),1);
        else
            idx_data_ahi = X_test.ahi < 15;
        end
        if covars == 1
            idx_pv = idx_pv_1;
        elseif covars == 2
            idx_pv = [idx_pv_1 idx_pv_2];
        elseif covars == 0
            idx_pv = idx_pv_0;
        end
        
        % Select data subset 
        idx_data = ismember(X_test.cohort_code, in_c) & X_test.v1 == 1 & idx_data_hyp & idx_data_ahi;
        
        % Data imputation
        x_data = X_test{:, [idx_rv idx_pv_aee idx_pv]};
        x_data_missing = X_test{idx_data, idx_pv};
        x_data_imp = x_data;
        if to_impute == 1 && covars > 0
            if k == 1
                if verbose_impute == 1
                    x_perc_1 = prctile(x_data_missing,[5, 50, 95]);
                    x_mean_1 = mean(x_data_missing,'omitnan');
                end
                writematrix(x_data_missing, "C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data\data_missing.csv");
                system('"C:\Program Files\R\R-4.0.4\bin\R.exe" CMD BATCH impute_data.R');
            end
            x_data_imputed = readmatrix("C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\data\data_imputed.csv");
            if verbose_impute == 1
                x_perc_2 = prctile(x_data_imputed(:,2:end),[5, 50, 95]);
                x_mean_2 = mean(x_data_imputed(:,2:end),'omitnan');
                x_change = [(x_mean_2 - x_mean_1)./(x_mean_1 + eps); (x_perc_2 - x_perc_1)./(x_perc_1 + eps)];
            end
            x_data_imp(idx_data, 3:end) = x_data_imputed(:,2:end);
        end
        idx_pv_all = [idx_pv_aee idx_pv];
        
        % Select data subset
        idx_data = ismember(X_test.cohort_code, in_c) & ~any(isnan(x_data_imp),2) & ~isnan(X_test.(mortality_type)) & X_test.v1 == 1 & idx_data_hyp & idx_data_ahi;
        idx_data_wbl = ismember(X_test.cohort_code, in_c) & ~any(isnan(x_data_imp),2) & (x_data_imp(:,1) ~= 0) & ~isnan(X_test.(mortality_type)) & X_test.v1 == 1 & idx_data_hyp & idx_data_ahi;
        
        % Variable type summary
        pv_std = std(x_data_imp(idx_data, 2:end));
        idx_binary = arrayfun(@(x) ~any(unique(x_data_imp(idx_data,x)) ~= 0 & unique(x_data_imp(idx_data,x)) ~= 1), 2:size(x_data_imp,2));
        pv_std(idx_binary) = 1;
        
        % Proportional Hazards model
        [b,logl,H,stats] = coxphfit(x_data_imp(idx_data,2:end),x_data_imp(idx_data,1),'Censoring',X_test.(mortality_type)(idx_data));
        [parmHat, parmCI] = wblfit(x_data_imp(idx_data_wbl,1), 0.05, X_test.(mortality_type)(idx_data_wbl));
        fprintf(['Cox proportional hazards regression. %s ~ %s', repmat(' + %s',1,length(idx_pv_all)-1), '.\n'],X_test.Properties.VariableNames{[idx_rv idx_pv_all]});
        fprintf('\t\tHR (CI)\tB\tp\n');
        for i = 1:length(idx_pv_all)
            CI_raw = (stats.beta(i) + [-1 1]*1.96*stats.se(i));
            CI = exp(pv_std(i) * CI_raw);
            fprintf('%s\t\t%.2f, (%.2f, %.2f)\t%.5f\t%.5g\n',X_test.Properties.VariableNames{idx_pv_all(i)},exp(pv_std(i)*stats.beta(i)),CI(1),CI(2),stats.beta(i),stats.p(i));
        end
        
        SAI_var_idx = contains(X_test.Properties.VariableNames(idx_pv_all), var_r{1});
        %         SAI_var_std = pv_std(SAI_var_idx);
        SAI_var_std = 10;
        SAI_var_beta = [stats.beta(SAI_var_idx) stats.beta(SAI_var_idx) + [-1 1]*1.96*stats.se(SAI_var_idx)];
        
        % LE
        for i = 1:length(LE_AGE)
            idx_age_interval = x_data_imp(:,3) >= (LE_AGE(i) - 10) & x_data_imp(:,3) <= (LE_AGE(i) + 10);
            idx_z_var = [2:(find(SAI_var_idx)) (find(SAI_var_idx)+2):size(x_data_imp,2)];
            %         zx = median((x_data_imp(idx_data_wbl,idx_z_var)-mean(x_data_imp(idx_data_wbl,idx_z_var))))*b(idx_z_var-1);
            zx = median(([repmat(LE_AGE(i), sum(idx_data_wbl & idx_age_interval), 1) x_data_imp(idx_data_wbl & idx_age_interval,idx_z_var(2:end))]-mean(x_data_imp(idx_data_wbl & idx_age_interval,idx_z_var))))*b(idx_z_var-1);
            sw = 1-wblcdf(0:((365*100)-1),parmHat(1),parmHat(2))';
            LE_p = sum(sw.^(exp((zx + SAI_var_std * SAI_var_beta)))) / 365;
            LE_n = sum(sw.^(exp((zx - SAI_var_std * SAI_var_beta)))) / 365;
            LE_diff = LE_p - LE_n;
            LE_ALL{k, covars+1, i} = LE_diff;
        end
%         h = figure;
%         h.Position(3:4) = [600 400];
%         centerfig(h);
%         hold all
%         plot_cell = cell(3,1);
%         plot_cell{1} = stairs(H(:,1)/365,exp(-H(:,2) * exp(-SAI_var_std * SAI_var_beta(1))),'b','LineWidth',2);
%         stairs(H(:,1)/365,exp(-H(:,2) * exp(-SAI_var_std * SAI_var_beta(2))),'--b','LineWidth',1);
%         stairs(H(:,1)/365,exp(-H(:,2) * exp(-SAI_var_std * SAI_var_beta(3))),'--b','LineWidth',1);
%         plot_cell{2} = stairs(H(:,1)/365,exp(-H(:,2) * exp(SAI_var_std * SAI_var_beta(1))),'r','LineWidth',2);
%         stairs(H(:,1)/365,exp(-H(:,2) * exp(SAI_var_std * SAI_var_beta(2))),'--r','LineWidth',1);
%         stairs(H(:,1)/365,exp(-H(:,2) * exp(SAI_var_std * SAI_var_beta(3))),'--r','LineWidth',1);
%         plot_cell{3} = plot(0,1,'--k');
%         grid minor
%         xlabel('Follow-up [years]')
%         ylabel('Survival %')
%         %         legend(horzcat(plot_cell{:}),{'-1 \sigma(AEE)','+1 \sigma(AEE)','95 % CI'},'Location','northeast')
%         legend(horzcat(plot_cell{:}),{'-10 AEE','+10 AEE','95 % CI'},'Location','northeast')
%         set(gcf,'Color',[1 1 1]);
%         set( findall(h, '-property', 'fontsize'), 'fontsize', 10);
%         export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\survival_curve_all_' names{k} '_model_' num2str(covars) '_mt_' mortality_type], '-pdf', '-transparent');
        
        HR_ALL{k,covars + 1} = exp(SAI_var_beta*SAI_var_std);
    end
end

for k = 1:size(HR_ALL,1)
    fprintf('%s\t%.2f, (%.2f, %.2f)\t%.2f, (%.2f, %.2f)\t%.2f, (%.2f, %.2f)\n',names{k},HR_ALL{k,:});
end

disp(vertcat(LE_ALL{9,:,:}));
% for i = 1:size(LE_AGE,1)
%     for k = 1:size(LE_ALL,1)
%         fprintf('%s\t%.2f, (%.2f, %.2f)\t%.2f, (%.2f, %.2f)\t%.2f, (%.2f, %.2f)\n',names{k},LE_ALL{k,:,i});
%     end
% end
%% Cox proportional hazards model - univariate (All data)
% Race
X_test.race_01 = double(X_test.race == 1);
X_test.race_01(isnan(X_test.race)) = nan;
% Smoking
X_test.smoke_02 = double(X_test.smoke == 2);
X_test.smoke_02(isnan(X_test.smoke)) = nan;
X_test.smoke_01 = double(X_test.smoke == 1);
X_test.smoke_01(isnan(X_test.smoke)) = nan;
% Cohort
X_test.wsc_cohort = double(X_test.cohort_code == 3);
X_test.wsc_cohort(isnan(X_test.cohort_code)) = nan;
X_test.shhs_cohort = double(X_test.cohort_code == 6);
X_test.shhs_cohort(isnan(X_test.cohort_code)) = nan;
X_test.mros_cohort = double(X_test.cohort_code == 4);
X_test.mros_cohort(isnan(X_test.cohort_code)) = nan;

% Mortality type:
mortality_type = 'vital';

% Cohorts
in_c = [3 4 6];

% Verbose imputation stats
idx_pv_0 = find(ismember(X_test.Properties.VariableNames,{'age','sex','bmi','wsc_cohort','shhs_cohort'}));
idx_pv_1 = find(ismember(X_test.Properties.VariableNames,{'race_01','smoke_01','edu_02','edu_03','edu_04','smoke_02','alch','hype','caf','ben','ant','remp','ari','ahi','waso','O280','n2p','ESS','t2d','haa','stro','chf','copd','ams','awaso','mmse','pase'}));

% Collect HR
HR_uni_all = cell(5,length(idx_pv_1));

% Select Model
% k = 3;idx_pv_1
for k = 1:length(idx_pv_1)
    
    % If variable - choose subset of cohorts
    if contains(X_test.Properties.VariableNames{idx_pv_1(k)},{'ams','awaso','mmse','pase'})
        idx_pv_0 = find(ismember(X_test.Properties.VariableNames,{'age','bmi'}));
    elseif contains(X_test.Properties.VariableNames{idx_pv_1(k)},'copd')
        idx_pv_0 = find(ismember(X_test.Properties.VariableNames,{'age','sex','bmi','shhs_cohort'}));
    else
        idx_pv_0 = find(ismember(X_test.Properties.VariableNames,{'age','sex','bmi','wsc_cohort','shhs_cohort'}));
    end
    
    % Response variable
    idx_rv = cellfun(@(x) find(ismember(X_test.Properties.VariableNames, x)), {'t_death'});
    
    % Predictors
    idx_pv = [idx_pv_0 idx_pv_1(k)];
    idx_pv_all = idx_pv;
    
    % Select data subset
    x_data = X_test{:, [idx_rv idx_pv]};
    x_data_imp = x_data;
    
    % Select data subset
    idx_data = ismember(X_test.cohort_code, in_c) & ~any(isnan(x_data_imp),2) & ~isnan(X_test.(mortality_type)) & X_test.v1 == 1;
    idx_data_wbl = ismember(X_test.cohort_code, in_c) & ~any(isnan(x_data_imp),2) & (x_data_imp(:,1) ~= 0) & ~isnan(X_test.(mortality_type)) & X_test.v1 == 1;
    
    % Variable type summary
    pv_std = std(x_data_imp(idx_data, 2:end));
    idx_binary = arrayfun(@(x) ~any(unique(x_data_imp(idx_data,x)) ~= 0 & unique(x_data_imp(idx_data,x)) ~= 1), 2:size(x_data_imp,2));
    pv_std(idx_binary) = 1;
    
    % Proportional Hazards model
    [b,logl,H,stats] = coxphfit(x_data_imp(idx_data,2:end),x_data_imp(idx_data,1),'Censoring',X_test.vital(idx_data));
    fprintf(['Cox proportional hazards regression. %s ~ %s', repmat(' + %s',1,length(idx_pv_all)-1), '.\n'],X_test.Properties.VariableNames{[idx_rv idx_pv_all]});
    fprintf('\t\tHR (CI)\tB\tp\n');
    for i = 1:length(idx_pv_all)
        CI_raw = (stats.beta(i) + [-1 1]*1.96*stats.se(i));
        CI = exp(pv_std(i) * CI_raw);
        fprintf('%s\t\t%.2f, (%.2f, %.2f)\t%.5f\t%.5g\n',X_test.Properties.VariableNames{idx_pv_all(i)},exp(pv_std(i)*stats.beta(i)),CI(1),CI(2),stats.beta(i),stats.p(i));
    end
    
    var_idx = contains(X_test.Properties.VariableNames(idx_pv_all), X_test.Properties.VariableNames(idx_pv_1(k)));
    var_std = pv_std(var_idx);
    var_beta = [stats.beta(var_idx) stats.beta(var_idx) + [-1 1]*1.96*stats.se(var_idx)];
    
    HR_uni_all{1,k} = sum(idx_data);
    HR_uni_all{2,k} = var_std;
    HR_uni_all{3,k} = exp(var_beta*var_std);
    HR_uni_all{4,k} = sum(X_test{idx_data,idx_pv_1(k)},'omitnan');
    HR_uni_all{5,k} = idx_binary(var_idx);
    
end

for k = 1:size(HR_uni_all,2)
    if HR_uni_all{5,k} == 1
        fprintf('%s\t%.0f / %.0f \t%.2f\t%.2f (%.2f, %.2f)\n',X_test.Properties.VariableNames{idx_pv_1(k)},HR_uni_all{4,k},HR_uni_all{1,k},HR_uni_all{2,k},HR_uni_all{3,k});
    else
        fprintf('%s\t%.0f \t%.2f\t%.2f (%.2f, %.2f)\n',X_test.Properties.VariableNames{idx_pv_1(k)},HR_uni_all{1,k},HR_uni_all{2,k},HR_uni_all{3,k});
    end
end
%% Correlation analysis TODO: AEE ~ 1 + age + sex + cohort + var
% k = 10;
in_c = 1:6;

var_r = {'ari','ahi','plmi','sl','tst','waso','se','n1p','n2p','n3p','remp','O280'};
for i = 1:length(var_r)
    X_test.([var_r{i} '_z']) = (X_test.(var_r{i}) - mean(X_test.(var_r{i}),'omitnan')) / std(X_test.(var_r{i}),'omitnan');
end
var_r = cellfun(@(x) [x '_z'],var_r,'Un',0);
idx_var_corr = cellfun(@(x) find(ismember(X_test.Properties.VariableNames, x)), var_r);
X_test.cohort_dummy = categorical(X_test.cohort);

b_age_lm_all = zeros(length(idx_var_corr), length(names));
p_age_lm_all = zeros(length(idx_var_corr), length(names));
n_age_lm_all = zeros(length(idx_var_corr), length(names));

for k = 1:length(names)
    
    b_age_lm = zeros(length(idx_var_corr),1);
    p_age_lm = zeros(length(idx_var_corr),1);
    n_age_lm = zeros(length(idx_var_corr),1);
    
    for i = 1:length(idx_var_corr)
        idx_pv = [find(ismember(X_test.Properties.VariableNames,{'age','sex','bmi','cohort_dummy'})) idx_var_corr(i)];
        idx_rv = find(ismember(X_test.Properties.VariableNames,['SAI_' names{k}]));
        idx_data = ~any(ismissing(X_test(:,[idx_rv idx_pv])),2) & ismember(X_test.cohort_code, in_c);
        mdl = fitlm(X_test(idx_data,:),'ResponseVar',idx_rv,'PredictorVars',idx_pv);
        n_age_lm(i,1) = size(mdl.Residuals,1);
        idx_mdl_g = ismember(mdl.Coefficients.Row, X_test.Properties.VariableNames(idx_var_corr(i)));
        b_age_lm(i,1) = mdl.Coefficients.Estimate(idx_mdl_g);
        p_age_lm(i,1) = mdl.Coefficients.pValue(idx_mdl_g);
    end
    
    T_b_age_lm = array2table(b_age_lm);
    T_p_age_lm = array2table(p_age_lm);
    T_b_age_lm.Properties.VariableNames = {'SAI_adj'};
    T_b_age_lm.Properties.RowNames = var_r;
    T_p_age_lm.Properties.VariableNames = {'SAI_adj'};
    T_p_age_lm.Properties.RowNames = var_r;
    
    p_age_lm_all(:,k) = p_age_lm(:,1);
    b_age_lm_all(:,k) = b_age_lm(:,1);
    n_age_lm_all(:,k) = n_age_lm(:,1);
end


fprintf(['Model\t\x3C3' repmat('\t%s',1,length(names)) '\n'],names{:});
for i = 1:size(p_age_lm_all)
    fprintf('%s (n = %d)',var_r{i},n_age_lm_all(i));
    fprintf('\t%.3g',std(X_test.(var_r{i}(1:end-2)),'omitnan'));
    for k = 1:length(names)
        sig_str = sprintf(' (p = %.2g)',p_age_lm_all(i,k));
        fprintf('\t%.2g %s',b_age_lm_all(i,k),sig_str);
    end
    fprintf('\n');
end

%% Correlation analysis TODO: AEE ~ 1 + age + sex + cohort + binary var
% k = 10;
in_c = 1:6;
X_test.bmi_z = (X_test.bmi - mean(X_test.bmi, 'omitnan'))/std(X_test.bmi,'omitnan');
var_r = {'sex','bmi_z','race_01','smoke_01','edu_02','edu_03','edu_04','smoke_02','hype','ben','ant','t2d','haa','stro','chf','copd'};
% for i = 1:length(var_r)
%     X_test.([var_r{i} '_z']) = (X_test.(var_r{i}) - mean(X_test.(var_r{i}),'omitnan')) / std(X_test.(var_r{i}),'omitnan');
% end
% var_r = cellfun(@(x) [x '_z'],var_r,'Un',0);
idx_var_corr = cellfun(@(x) find(ismember(X_test.Properties.VariableNames, x)), var_r);
X_test.cohort_dummy = categorical(X_test.cohort);

b_age_lm_all = zeros(length(idx_var_corr), length(names));
p_age_lm_all = zeros(length(idx_var_corr), length(names));
n_age_lm_all = zeros(length(idx_var_corr), length(names));
s_age_lm_all = zeros(length(idx_var_corr), length(names));

for k = 1:length(names)
    
    b_age_lm = zeros(length(idx_var_corr),1);
    p_age_lm = zeros(length(idx_var_corr),1);
    n_age_lm = zeros(length(idx_var_corr),1);
    s_age_lm = zeros(length(idx_var_corr),1);
    
    for i = 1:length(idx_var_corr)
        idx_pv = [find(ismember(X_test.Properties.VariableNames,{'age','sex','bmi_z','cohort_dummy'})) idx_var_corr(i)];
        idx_rv = find(ismember(X_test.Properties.VariableNames,['SAI_' names{k}]));
        idx_data = ~any(ismissing(X_test(:,[idx_rv idx_pv])),2) & ismember(X_test.cohort_code, in_c);
        mdl = fitlm(X_test(idx_data,:),'ResponseVar',idx_rv,'PredictorVars',idx_pv);
        n_age_lm(i,1) = size(mdl.Residuals,1);
        s_age_lm(i,1) = sum(X_test{idx_data, idx_var_corr(i)});
        idx_mdl_g = ismember(mdl.Coefficients.Row, X_test.Properties.VariableNames(idx_var_corr(i)));
        b_age_lm(i,1) = mdl.Coefficients.Estimate(idx_mdl_g);
        p_age_lm(i,1) = mdl.Coefficients.pValue(idx_mdl_g);
    end
    
    T_b_age_lm = array2table(b_age_lm);
    T_p_age_lm = array2table(p_age_lm);
    T_b_age_lm.Properties.VariableNames = {'SAI_adj'};
    T_b_age_lm.Properties.RowNames = var_r;
    T_p_age_lm.Properties.VariableNames = {'SAI_adj'};
    T_p_age_lm.Properties.RowNames = var_r;
    
    p_age_lm_all(:,k) = p_age_lm(:,1);
    b_age_lm_all(:,k) = b_age_lm(:,1);
    n_age_lm_all(:,k) = n_age_lm(:,1);
    s_age_lm_all(:,k) = s_age_lm(:,1);
end


fprintf(['Model\tsum' repmat('\t%s',1,length(names)) '\n'],names{:});
for i = 1:size(p_age_lm_all)
    fprintf('%s (n = %d)',var_r{i},n_age_lm_all(i));
    fprintf('\t%d',s_age_lm_all(i));
    for k = 1:length(names)
        sig_str = sprintf(' (p = %.2g)',p_age_lm_all(i,k));
        fprintf('\t%.2g %s',b_age_lm_all(i,k),sig_str);
    end
    fprintf('\n');
end

%% High-level alpha correlation to sleep stages
%  Interpretability subset of data
rng(260794);
int_idx = arrayfun(@(x) randsample(find(X_test.cohort_code == x), 10), 1:max(X_test.cohort_code),'Un',0);
int_idx = cell2mat(int_idx');

k = 8;
alpha_ssc_dist = zeros(60,5);
n = size(int_idx,1);

for i = 1:n
    record = X_test.names{int_idx(i)};
    %     record_2 = dir([paths{k} '\predictions_pre_sep\' record '*']);
    %     record_2 = record_2.name(1:end-5);
    cohort_code = X_test.cohort_code(int_idx(i));
    if i == 1
        [preds, ~] = read_model_alpha(paths{k}, [], false, true);
    end
    [alpha, ~] = match_record_alpha(preds, [], record, cohort_code);
    h5_path = dir(['H:\nAge\all\' record '*']);
    ssc = h5read([h5_path.folder '\' h5_path.name],'/SSC');
    ssc = ssc(1:min(length(ssc),(120*(60*5)/30)));
    alpha_ssc_dist(i,:) = arrayfun(@(x) mean(alpha(ceil(find(ssc == x) * (30) / (60 * 5)))), 1:-1:-3);
end

disp(mean(alpha_ssc_dist,'omitnan'));
% disp(std(alpha_ssc_dist,'omitnan'));

%% Low-level alpha correlation to eeg frequency magnitude and magnitude change
%  Interpretability subset of data
% rng(260794);
% int_idx = arrayfun(@(x) randsample(find(X_test.cohort_code == x), 10), 1:max(X_test.cohort_code),'Un',0);
% int_idx = cell2mat(int_idx');
%
% k = 5;
% n = size(int_idx,1);
% alpha_ps1_ssc = cell(n,7);
% ps1_ssc = cell(n,7);
% alpha_ps1_difp_ssc = cell(n,7);
% ps1_difp_ssc = cell(n,7);
% alpha_ps1_difn_ssc = cell(n,7);
% ps1_difn_ssc = cell(n,7);
% fs = 128;
%
% for i = 1:n
%     record = X_test.names{int_idx(i)};
%     record_2 = dir([paths{k} '\predictions_pre_sep\' record '*']);
%     record_2 = record_2.name(1:end-5);
%     cohort_code = X_test.cohort_code(int_idx(i));
%     if i == 1
%         [preds, preds_pre] = read_model_alpha(paths{k}, record_2, false, false);
%     else
%         [~, preds_pre] = read_model_alpha(paths{k}, record_2, true, false);
%     end
%     [alpha, alpha_pre] = match_record_alpha(preds, preds_pre, record, cohort_code);
%     h5_path = dir(['H:\nAge\all\' record '*']);
%     ssc = h5read([h5_path.folder '\' h5_path.name],'/SSC');
%     psg = h5read([h5_path.folder '\' h5_path.name],'/PSG');
%     alpha_pre_2 = reshape(alpha_pre',1,[]);
%     eeg1 = psg(:,1);
%     eeg2 = psg(:,2);
%     [~,f1,t1,ps1] = spectrogram(eeg1,7.5*fs,5*fs,0:0.1:fs/2,fs,'yaxis');
%     [~,f2,t2,ps2] = spectrogram(eeg2,7.5*fs,5*fs,0:0.1:fs/2,fs,'yaxis');
%     ps1 = (ps1 + ps2)/2;
%     ps1_crop = ([zeros(length(f1),1) ps1(:,1:(length(alpha_pre_2) - 2)), zeros(length(f1),1)]);
%     %     ps1_crop(ps1_crop < -70) = -70;
%     ps1_crop_dif = [zeros(length(f1),2) ps1_crop(:,3:end)-ps1_crop(:,1:end-2)];
%     ps1_crop_difp = ps1_crop_dif .* (ps1_crop_dif > 0);
%     ps1_crop_difn = ps1_crop_dif .* (ps1_crop_dif < 0);
%     alpha_ps1_difp = ps1_crop_difp .* repmat(alpha_pre_2,length(f1),1);
%     alpha_ps1_difn = ps1_crop_difn .* repmat(alpha_pre_2,length(f1),1);
%     alpha_ps1 = ps1_crop .* repmat(alpha_pre_2,length(f1),1);
%     if length(ssc) > length(alpha_pre_2)*2.5/30
%         ssc = ssc(1:length(alpha_pre_2)*2.5/30)';
%     end
%     ssc = ssc(:);
%     ssc_up = repelem(ssc',30/2.5);
%     alpha_ps1_ssc(i,1:5) = arrayfun(@(x) mean(alpha_ps1(:,ssc_up == x),2),1:-1:-3,'Un',0);
%     ps1_ssc(i,1:5) = arrayfun(@(x) mean(ps1_crop(:,ssc_up == x),2),1:-1:-3,'Un',0);
%     alpha_ps1_ssc(i,6) = {mean(alpha_ps1(:,ssc_up < 1),2)};
%     ps1_ssc(i,6) = {mean(ps1_crop(:,ssc_up < 1),2)};
%     alpha_ps1_ssc(i,7) = {mean(alpha_ps1,2)};
%     ps1_ssc(i,7) = {mean(ps1_crop,2)};
%
%     idx = false(1,size(ssc_up,2));
%     idx(unique((2:(5*60/2.5)) + [0:120:length(ssc_up)]')) = true;
%     idx = idx(1:size(ssc_up,2));
%     alpha_ps1_difp_ssc(i,1:5) = arrayfun(@(x) mean(alpha_ps1_difp(:,ssc_up == x & idx),2),1:-1:-3,'Un',0);
%     ps1_difp_ssc(i,1:5) = arrayfun(@(x) mean(ps1_crop_difp(:,ssc_up == x & idx),2),1:-1:-3,'Un',0);
%     alpha_ps1_difn_ssc(i,1:5) = arrayfun(@(x) mean(alpha_ps1_difn(:,ssc_up == x & idx),2),1:-1:-3,'Un',0);
%     ps1_difn_ssc(i,1:5) = arrayfun(@(x) mean(ps1_crop_difn(:,ssc_up == x & idx),2),1:-1:-3,'Un',0);
%     alpha_ps1_difp_ssc(i,6) = {mean(alpha_ps1_difp(:,idx & ssc_up < 1),2)};
%     ps1_difp_ssc(i,6) = {mean(ps1_crop_difp(:,idx & ssc_up < 1),2)};
%     alpha_ps1_difn_ssc(i,6) = {mean(alpha_ps1_difn(:,idx & ssc_up < 1),2)};
%     ps1_difn_ssc(i,6) = {mean(ps1_crop_difn(:,idx & ssc_up < 1),2)};
%     alpha_ps1_difp_ssc(i,7) = {mean(alpha_ps1_difp(:,idx),2)};
%     ps1_difp_ssc(i,7) = {mean(ps1_crop_difp(:,idx),2)};
%     alpha_ps1_difn_ssc(i,7) = {mean(alpha_ps1_difn(:,idx),2)};
%     ps1_difn_ssc(i,7) = {mean(ps1_crop_difn(:,idx),2)};
%
% end
%
% figure;
% for i = 1:6
%     subplot(3,2,i)
%     baseline_spec = mean(cell2mat(ps1_ssc(:,i)'),2,'omitnan');
%     alpha_w_spec  = mean(cell2mat(alpha_ps1_ssc(:,i)'),2,'omitnan');
%     baseline_spec_n = baseline_spec * (sum(alpha_w_spec) / sum(baseline_spec));
%     plot(f1, 10*log10(alpha_w_spec ./ baseline_spec_n));
%     hold on
%     plot(f1([1 end]), [0 0],'--k')
%     xlabel('Frequency [Hz]')
%     grid minor
%     %ylabel('Relative EEG power of \alpha-weighed spectrogram')
% end
%
% figure;
% for i = 1:6
%     subplot(3,2,i)
%
%     baseline_spec = mean(cell2mat(ps1_difp_ssc(:,i)'),2,'omitnan');
%     alpha_w_spec  = mean(cell2mat(alpha_ps1_difp_ssc(:,i)'),2,'omitnan');
%     baseline_spec_n = baseline_spec * (sum(alpha_w_spec) / sum(baseline_spec));
%     plot(f1, 10*log10(alpha_w_spec ./ baseline_spec_n));
%     hold on
%     plot(f1([1 end]), [0 0],'--k')
%     xlabel('Frequency [Hz]')
%     grid minor
% end
%
% figure;
% for i = 1:6
%     subplot(3,2,i)
%
%     baseline_spec = mean(cell2mat(ps1_difn_ssc(:,i)'),2,'omitnan');
%     alpha_w_spec  = mean(cell2mat(alpha_ps1_difn_ssc(:,i)'),2,'omitnan');
%     baseline_spec_n = baseline_spec * (sum(alpha_w_spec) / sum(baseline_spec));
%     plot(f1, 10*log10(alpha_w_spec ./ baseline_spec_n));
%     hold on
%     plot(f1([1 end]), [0 0],'--k')
%     xlabel('Frequency [Hz]')
%     grid minor
% end

%% Low-level interpretability correlation to eeg frequency magnitude and magnitude change
%  Interpretability subset of data
%  1) Average in sleep stages, arousals, and apneas/hyponeas
%  2) Timelock to sleep stage transitions
%  (W-NREM, NREM-W, W-REM, REM-W)
%  3) Timelock to arousal and apnea/hypopnea onset and offset
%  4) Average relevance at frequency bands and shifts

% TODO: Change dataset to 40 CFS + 40 MrOS + 40 SHHS
% rng(260794);
% int_idx_120 = arrayfun(@(x) randsample(find(X_test.cohort_code == x & ~sum(1:size(X_test,1) == int_idx)'), 30), [1 4 6],'Un',0);
% int_idx_120 = cell2mat(int_idx_120');
% int_idx_120 = [int_idx([1:10, 31:40, 51:60]); int_idx_120];
% load('int_idx_120');
% int_idx = int_idx_120;
% int_idx_record = X_test.names(int_idx);
load('int_idx_record');
% Data
% rng(260794);
% int_idx = arrayfun(@(x) randsample(find(X_test.cohort_code == x), 10), 1:6,'Un',0);
% int_idx = cell2mat(int_idx');
fs = 128;

% Model
k = 7;
SaO2_mode = 2;
% SaO2_mode_str = '_sao2';
% SaO2_mode_str = '_no_sao2';
SaO2_mode_str = '_all';

% Relevance method
rel_method = 'int_grad';

% result variable pre-allocation
n = size(int_idx,1);
int_ssc_avg = zeros(n,5); % (1)
int_ar_avg = zeros(n,2); % (1)
int_ah_avg = zeros(n,2); % (1)
int_od_avg = zeros(n,2); % (1)
ssc_trans_leg = {'NREM-W','REM-W','N2-N1','N3-N2','N2-N3','N1-N2','NREM-REM','REM-NREM'};
ssc_trans = {{1,-3:-1},{1,0},{-1,-2},{-2,-3},{-3,-2},{-2,-1},{0,-3:-1},{-3:-1,0}};
tl_window = 70*fs;
int_tl_ssc_trans = cell(n,6); % (2)
int_tl_ar = cell(n,2); % (3)
int_tl_ah = cell(n,2); % (3)
int_tl_od = cell(n,2); % (3)
int_ps1_ssc = cell(n,6,2);
ps1_ssc = cell(n,6);
int_ps1_difp_ssc = cell(n,6,2);
ps1_difp_ssc = cell(n,6);
int_ps1_difn_ssc = cell(n,6,2);
ps1_difn_ssc = cell(n,6);

for i = 1:n
    % Declare paths
    %     record = X_test.names{int_idx(i)};
    record = int_idx_record{i};
    record_2 = dir([paths{k} '\predictions_pre_sep\' record '*']);
    record_2 = record_2.name(1:end-5);
    cohort_code = X_test.cohort_code(int_idx(i));
    h5_path = dir([paths{k} '\interpretation\' rel_method '\' record '*']);
    % Read model interpretation
    interp = h5read([h5_path.folder '\' h5_path.name],'/Interpretation');
    if k == 7
        if SaO2_mode == 1
            interp = interp(:,end);
        elseif SaO2_mode == 0
            interp = interp(:,1:end-4);
        end
    end
    interp = mean(interp,2);
    interp_avg = mean(reshape(interp,fs,[]),1);
    interp_avg_p = interp_avg .* (interp_avg > 0);
    interp_avg_n = interp_avg .* (interp_avg < 0);
    % Smooth interp
    g_std = 30;
    g_win = gausswin(fs * 10, (fs * 10)/(g_std * 2));
    rel_g = conv2(interp, g_win, 'same') / sum(g_win);
    % Read sleep staging and PSG
    h5_path = dir(['H:\nAge\all\' record '*']);
    ssc = h5read([h5_path.folder '\' h5_path.name],'/SSC');
    %     psg = h5read([h5_path.folder '\' h5_path.name],'/PSG');
    %     if length(ssc) > (size(psg,1)/(fs*30))
    %         ssc = ssc(1:size(psg,1)/(fs*30));
    %     end
    if length(ssc) > (size(interp,1)/(fs*30))
        ssc = ssc(1:size(interp,1)/(fs*30));
    end
    ssc_up = repelem(ssc',30*fs);
    % Load arousal and apneas
    if contains(record,'cfs')
        ftype = 'cfs';
        evt_path = ['G:\cfs\polysomnography\annotations-events-profusion\' record '-profusion.xml'];
    elseif contains(record,'SSC')
        ftype = 'ssc';
        evt_path = ['G:\ssc\polysomnography\labels\' record '.evts'];
    elseif contains(record,'shhs')
        ftype = 'shhs';
        evt_path = ['H:\shhs\polysomnography\annotations-events-profusion\shhs1\' record '-profusion.xml'];
    elseif contains(record,'EDFAndScore')
        ftype = 'stages';
        evt_path = '';
    elseif contains(record, 'mros')
        ftype = 'mros';
        evt_path = ['G:\mros\polysomnography\annotations-events-profusion\visit1\' record '-profusion.xml'];
    else
        ftype = 'wsc';
        evt_path = '';
    end
    L = round(length(rel_g)/fs);
    [ar,ar_seq] = LoadAR(evt_path,L,ftype);
    ar_seq_up = repelem(ar_seq,fs);
    [ah,ah_seq] = LoadAH(evt_path,L,ftype);
    ah_seq_up = repelem(ah_seq,fs);
    [od,od_seq] = LoadOD(evt_path,L,ftype);
    od_seq_up = repelem(od_seq,fs);
    % 1) Average in sleep stages, arousals, and apneas/hyponeas
    int_ssc_avg(i,:) = arrayfun(@(x) mean(rel_g(ssc_up == x)),1:-1:-3);
    int_ar_avg(i,:) = arrayfun(@(x) mean(rel_g(ar_seq_up == x)),[0, 1]);
    int_ah_avg(i,:) = arrayfun(@(x) mean(rel_g(ah_seq_up == x)),[0, 1]);
    int_od_avg(i,:) = arrayfun(@(x) mean(rel_g(od_seq_up == x)),[0, 1]);
    %  2) Timelock to sleep stage transitions
    idx_ssc_trans = cellfun(@(x) 1+find(ismember(ssc_up(2:end),x{1}) & ismember(ssc_up(1:end-1),x{2})),ssc_trans,'Un',0);
    idx_ssc_trans = cellfun(@(x) x(rem((x-1),5*60*fs) ~= 0 & x <= L*fs-tl_window & x >= tl_window),idx_ssc_trans,'Un',0);
    for j = 1:length(idx_ssc_trans)
        idx_ssc_trans_j = idx_ssc_trans{j}(rem((idx_ssc_trans{j}-1),5*60*fs) ~= 0);
        int_tl_ssc_trans(i,j) = {mean(cell2mat(arrayfun(@(x) rel_g((x-tl_window+1):(x+tl_window)),idx_ssc_trans_j,'Un',0)),2)};
    end
    %  3) Timelock to arousal and apnea/hypopnea onset and offset
    if contains(ftype,{'cfs','mros','shhs'})
        idx_ar_onset = round(ar.start * fs);
        idx_ar_offset = round(ar.stop * fs);
        idx_ar_onset_2 = idx_ar_onset(rem(idx_ar_onset, 5*60*fs) >= 30*fs & rem(idx_ar_offset, 5*60*fs) <= (5*60-30)*fs & idx_ar_offset <= L*fs-tl_window & idx_ar_onset >= tl_window);
        idx_ar_offset_2 = idx_ar_offset(rem(idx_ar_onset, 5*60*fs) >= 30*fs & rem(idx_ar_offset, 5*60*fs) <= (5*60-30)*fs & idx_ar_offset <= L*fs-tl_window & idx_ar_onset >= tl_window);
        int_tl_ar(i,1) = {mean(cell2mat(arrayfun(@(x) rel_g((x-tl_window+1):(x+tl_window)),idx_ar_onset_2,'Un',0)),2)};
        int_tl_ar(i,2) = {mean(cell2mat(arrayfun(@(x) rel_g((x-tl_window+1):(x+tl_window)),idx_ar_offset_2,'Un',0)),2)};
        
        idx_ah_onset = round(ah.start * fs);
        idx_ah_offset = round(ah.stop * fs);
        idx_ah_onset_2 = idx_ah_onset(rem(idx_ah_onset, 5*60*fs) >= 30*fs & rem(idx_ah_offset, 5*60*fs) <= (5*60-30)*fs & idx_ah_offset <= L*fs-tl_window & idx_ah_onset >= tl_window);
        idx_ah_offset_2 = idx_ah_offset(rem(idx_ah_onset, 5*60*fs) >= 30*fs & rem(idx_ah_offset, 5*60*fs) <= (5*60-30)*fs & idx_ah_offset <= L*fs-tl_window & idx_ah_onset >= tl_window);
        int_tl_ah(i,1) = {mean(cell2mat(arrayfun(@(x) rel_g((x-tl_window+1):(x+tl_window)),idx_ah_onset_2,'Un',0)),2)};
        int_tl_ah(i,2) = {mean(cell2mat(arrayfun(@(x) rel_g((x-tl_window+1):(x+tl_window)),idx_ah_offset_2,'Un',0)),2)};
        
        idx_od_onset = round(od.start * fs);
        idx_od_offset = round(od.stop * fs);
        idx_od_onset_2 = idx_od_onset(rem(idx_od_onset, 5*60*fs) >= 30*fs & rem(idx_od_offset, 5*60*fs) <= (5*60-30)*fs & idx_od_offset <= L*fs-tl_window & idx_od_onset >= tl_window);
        idx_od_offset_2 = idx_od_offset(rem(idx_od_onset, 5*60*fs) >= 30*fs & rem(idx_od_offset, 5*60*fs) <= (5*60-30)*fs & idx_od_offset <= L*fs-tl_window & idx_od_onset >= tl_window);
        int_tl_od(i,1) = {mean(cell2mat(arrayfun(@(x) rel_g((x-tl_window+1):(x+tl_window)),idx_od_onset_2,'Un',0)),2)};
        int_tl_od(i,2) = {mean(cell2mat(arrayfun(@(x) rel_g((x-tl_window+1):(x+tl_window)),idx_od_offset_2,'Un',0)),2)};
    end
    % Spectrogram
    %     eeg1 = psg(:,1);
    %     eeg2 = psg(:,2);
    %     [~,f1,t1,ps1] = spectrogram(eeg1,3*fs,2*fs,0:0.1:fs/2,fs,'yaxis');
    %     [~,f2,t2,ps2] = spectrogram(eeg2,3*fs,2*fs,0:0.1:fs/2,fs,'yaxis');
    %     ps1 = (ps1 + ps2)/2;
    %     ps1_crop = ([zeros(length(f1),1) ps1(:,1:(length(interp_avg) - 2)), zeros(length(f1),1)]);
    %     % Change in spectrogram
    %     ps1_crop_dif = [zeros(length(f1),2) ps1_crop(:,3:end)-ps1_crop(:,1:end-2)];
    %     ps1_crop_difp = ps1_crop_dif .* (ps1_crop_dif > 0);
    %     ps1_crop_difn = ps1_crop_dif .* (ps1_crop_dif < 0);
    %     % Interpretation * spectrogram
    %     intp_ps1_difp = ps1_crop_difp .* repmat(interp_avg_p,length(f1),1);
    %     intp_ps1_difn = ps1_crop_difn .* repmat(interp_avg_p,length(f1),1);
    %     intn_ps1_difp = ps1_crop_difp .* repmat(interp_avg_n,length(f1),1);
    %     intn_ps1_difn = ps1_crop_difn .* repmat(interp_avg_n,length(f1),1);
    %     intp_ps1 = ps1_crop .* repmat(interp_avg_p,length(f1),1);
    %     intn_ps1 = ps1_crop .* repmat(interp_avg_n,length(f1),1);
    %     % Average over sleep stages
    %     ssc_up = repelem(ssc',30);
    %     int_ps1_ssc(i,1:5,1) = arrayfun(@(x) mean(intp_ps1(:,ssc_up == x),2),1:-1:-3,'Un',0);
    %     int_ps1_ssc(i,1:5,2) = arrayfun(@(x) mean(intn_ps1(:,ssc_up == x),2),1:-1:-3,'Un',0);
    %     ps1_ssc(i,1:5) = arrayfun(@(x) mean(ps1_crop(:,ssc_up == x),2),1:-1:-3,'Un',0);
    %     int_ps1_ssc(i,6,1) = {mean(intp_ps1,2)};
    %     int_ps1_ssc(i,6,2) = {mean(intn_ps1,2)};
    %     ps1_ssc(i,6) = {mean(ps1_crop,2)};
    %
    %     idx = false(1,size(ssc_up,2));
    %     idx(unique((2:(5*60)) + [0:(5*60):length(ssc_up)]')) = true;
    %     idx = idx(1:size(ssc_up,2));
    %     int_ps1_difp_ssc(i,1:5,1) = arrayfun(@(x) mean(intp_ps1_difp(:,ssc_up == x & idx),2),1:-1:-3,'Un',0);
    %     int_ps1_difp_ssc(i,1:5,2) = arrayfun(@(x) mean(intn_ps1_difp(:,ssc_up == x & idx),2),1:-1:-3,'Un',0);
    %     int_ps1_difn_ssc(i,1:5,1) = arrayfun(@(x) mean(intp_ps1_difn(:,ssc_up == x & idx),2),1:-1:-3,'Un',0);
    %     int_ps1_difn_ssc(i,1:5,2) = arrayfun(@(x) mean(intn_ps1_difn(:,ssc_up == x & idx),2),1:-1:-3,'Un',0);
    %     ps1_difp_ssc(i,1:5) = arrayfun(@(x) mean(ps1_crop_difp(:,ssc_up == x & idx),2),1:-1:-3,'Un',0);
    %     ps1_difn_ssc(i,1:5) = arrayfun(@(x) mean(ps1_crop_difn(:,ssc_up == x & idx),2),1:-1:-3,'Un',0);
    %     int_ps1_difp_ssc(i,6,1) = {mean(intp_ps1_difp(:,idx),2)};
    %     int_ps1_difp_ssc(i,6,2) = {mean(intn_ps1_difp(:,idx),2)};
    %     int_ps1_difn_ssc(i,6,1) = {mean(intp_ps1_difn(:,idx),2)};
    %     int_ps1_difn_ssc(i,6,2) = {mean(intn_ps1_difn(:,idx),2)};
    %     ps1_difp_ssc(i,6) = {mean(ps1_crop_difp(:,idx),2)};
    %     ps1_difn_ssc(i,6) = {mean(ps1_crop_difn(:,idx),2)};
end

names_plot = {'1','10','(a) Full Montage','(b) Central EEG','(c) EEG','(d) ECG','(e) Respiratory','(f) Ensemble - z','(g) Ensemble - Avg.','(h) Ensemble - Avg. EEG'};

g_win = gausswin(fs * 10, (fs * 10)/(g_std * 2));
leg = {'Arousal Onset','Arousal Offset'};
h = figure;
h.Position(3:4) = [800 300];
centerfig(h);
for i = 1:size(int_tl_ar,2)
    subplot(1,size(int_tl_ar,2),i)
    int_tl_ar_avg = int_tl_ar(cellfun(@(x) ~isempty(x), int_tl_ar(:,i)),i);
    int_tl_ar_avg = mean(cell2mat(int_tl_ar_avg'),2);
    int_tl_ar_avg_filt = conv(int_tl_ar_avg,g_win,'same')/sum(g_win);
    %     plot(((-tl_window+1):tl_window)/fs, int_tl_ar_avg);
    hold all
    plot(((-tl_window+1):tl_window)/fs, int_tl_ar_avg_filt * 10^4);
    yl = get(gca,'YLim');
    plot([0 0], yl,'--k')
    set(gca,'YLim',yl);
    grid minor
    xlim([-60 60]);
    title(leg{i});
    xlabel('Time [s]');
    if i == 1
        ylabel({[names_plot{k}],'Avg. Relevance 10^{-4}'});
    end
end
set(gcf,'Color',[1 1 1]);
set( findall(h, '-property', 'fontsize'), 'fontsize', 10);
export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\arousal_tl_relevance_g_' names{k} SaO2_mode_str], '-pdf', '-transparent');


% g_win = gausswin(fs*10);
leg = {'Apnea/Hypopnea Onset','Apnea/Hypopnea Offset'};
h = figure;
h.Position(3:4) = [800 300];
centerfig(h);
for i = 1:size(int_tl_ah,2)
    subplot(1,size(int_tl_ah,2),i)
    int_tl_ah_avg = int_tl_ah(cellfun(@(x) ~isempty(x), int_tl_ah(:,i)),i);
    int_tl_ah_avg = mean(cell2mat(int_tl_ah_avg'),2);
    int_tl_ah_avg_filt = conv(int_tl_ah_avg,g_win,'same')/sum(g_win);
    %     plot(((-tl_window+1):tl_window)/fs, int_tl_ah_avg);
    hold all
    plot(((-tl_window+1):tl_window)/fs, int_tl_ah_avg_filt * 10^4);
    yl = get(gca,'YLim');
    plot([0 0], yl,'--k')
    set(gca,'YLim',yl);
    grid minor
    xlim([-60 60]);
    title(leg{i});
    xlabel('Time [s]');
    if i == 1
        ylabel({[names_plot{k}],'Avg. Relevance 10^{-4}'});
    end
end
set(gcf,'Color',[1 1 1]);
set( findall(h, '-property', 'fontsize'), 'fontsize', 10);
export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\apnea_hypopnea_tl_relevance_g_' names{k} SaO2_mode_str], '-pdf', '-transparent');


% g_win = gausswin(fs*10);
leg = {'Desaturation Onset','Desaturation Offset'};
h = figure;
h.Position(3:4) = [800 300];
centerfig(h);
for i = 1:size(int_tl_ah,2)
    subplot(1,size(int_tl_od,2),i)
    int_tl_od_avg = int_tl_od(cellfun(@(x) ~isempty(x), int_tl_od(:,i)),i);
    int_tl_od_avg = mean(cell2mat(int_tl_od_avg'),2);
    int_tl_od_avg_filt = conv(int_tl_od_avg,g_win,'same')/sum(g_win);
    %     plot(((-tl_window+1):tl_window)/fs, int_tl_ah_avg);
    hold all
    plot(((-tl_window+1):tl_window)/fs, int_tl_od_avg_filt * 10^4);
    yl = get(gca,'YLim');
    plot([0 0], yl,'--k')
    set(gca,'YLim',yl);
    grid minor
    xlim([-60 60]);
    title(leg{i});
    xlabel('Time [s]');
    if i == 1
        ylabel({[names_plot{k}],'Avg. Relevance 10^{-4}'});
    end
end
set(gcf,'Color',[1 1 1]);
set( findall(h, '-property', 'fontsize'), 'fontsize', 10);
export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\desaturation_tl_relevance_g_' names{k} SaO2_mode_str], '-pdf', '-transparent');

h = figure;
h.Position(3:4) = [800 300];
centerfig(h);
for i = 1:size(int_tl_ssc_trans,2)
    subplot(2,size(int_tl_ssc_trans,2)/2,i)
    int_tl_ssc_trans_avg = int_tl_ssc_trans(cellfun(@(x) ~isempty(x), int_tl_ssc_trans(:,i)),i);
    int_tl_ssc_trans_avg = mean(cell2mat(int_tl_ssc_trans_avg'),2);
    int_tl_ssc_trans_avg_filt = filter(g_win,1,int_tl_ssc_trans_avg)/sum(g_win);
    %     plot(((-tl_window+1):tl_window)/fs, int_tl_ssc_trans_avg);
    hold all
    plot(((-tl_window+1):tl_window)/fs, int_tl_ssc_trans_avg_filt * 10^4);
    yl = get(gca,'YLim');
    plot([0 0], yl,'--k')
    set(gca,'YLim',yl);
    grid minor
    xlim([-30 30]);
    title(ssc_trans_leg{i});
    xlabel('Time [s]');
    if i == 1 || i == 5
        ylabel({[names_plot{k}],'Avg. Relevance 10^{-4}'});
    end
end
set(gcf,'Color',[1 1 1]);
set( findall(h, '-property', 'fontsize'), 'fontsize', 10);
export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\sleep_stage_transition_tl_relevance_g_' names{k} SaO2_mode_str], '-pdf', '-transparent');

% leg = {'W','REM','N1','N2','N3','All'};
% % Frequencies related to relevance
% % baseline,
% h = figure;
% h.Position(3:4) = [800 600];
% centerfig(h);
% for j = 1:2
%     for i = 1:5
%         subplot(2,5,i + (j-1)*5)
%         baseline_spec = abs(mean(cell2mat(ps1_ssc(:,i)'),2,'omitnan'));
%         int_w_spec  = abs(mean(cell2mat(int_ps1_ssc(:,i,j)'),2,'omitnan'));
%         baseline_spec_n = baseline_spec * (sum(int_w_spec) / sum(baseline_spec));
%         plot(f1, 10*log10(int_w_spec ./ baseline_spec_n));
%         hold on
%         plot(f1([1 end]), [0 0],'--k')
%         grid minor
%         xlim([0 45]);
%         if j == 1
%             title(leg{i});
%         else
%             xlabel('Frequency [Hz]');
%         end
%         if i == 1
%             ylabel('Relative power [dB]');
%         end
%     end
% end
%
% % Frequency changes (positive) related to relevance
% % baseline,
%
% h = figure;
% h.Position(3:4) = [800 600];
% centerfig(h);
% for j = 1:2
%     for i = 1:5
%         subplot(2,5,i + (j-1)*5)
%         baseline_spec = abs(mean(cell2mat(ps1_difp_ssc(:,i)'),2,'omitnan'));
%         int_w_spec  = abs(mean(cell2mat(int_ps1_difp_ssc(:,i,j)'),2,'omitnan'));
%         baseline_spec_n = baseline_spec * (sum(int_w_spec) / sum(baseline_spec));
%         plot(f1, 10*log10(int_w_spec ./ baseline_spec_n));
%         hold on
%         plot(f1([1 end]), [0 0],'--k')
%         grid minor
%         xlim([0 45]);
%         if j == 1
%             title(leg{i});
%         else
%             xlabel('Frequency [Hz]');
%         end
%         if i == 1
%             ylabel('Relative power [dB]');
%         end
%     end
% end

% figure;
% for i = 1:6
% subplot(3,2,i)
% plot(f1,( mean(cell2mat(ps1_dif_ssc(:,i)'),2,'omitnan')* sum(mean(cell2mat(int_ps1_dif_ssc(:,i)'),2,'omitnan')) / sum(mean(cell2mat(ps1_dif_ssc(:,i)'),2,'omitnan'))))
% hold on
% plot(f1,(mean(cell2mat(int_ps1_dif_ssc(:,i)'),2,'omitnan') ))
% end

%% Plot variables
d_age = 5;
age_ranges = 20:d_age:85;
vital_bmi_age_m_1 = arrayfun(@(x) mean(X_test.bmi(X_test.vital == 1 & X_test.age > x & X_test.age < x+d_age),'omitnan'),age_ranges);
vital_bmi_age_m_0 = arrayfun(@(x) mean(X_test.bmi(X_test.vital == 0 & X_test.age > x & X_test.age < x+d_age),'omitnan'),age_ranges);
vital_bmi_age_p5_1 = arrayfun(@(x) prctile(X_test.bmi(X_test.vital == 1 & X_test.age > x & X_test.age < x+d_age),5),age_ranges);
vital_bmi_age_p5_0 = arrayfun(@(x) prctile(X_test.bmi(X_test.vital == 0 & X_test.age > x & X_test.age < x+d_age),5),age_ranges);
vital_bmi_age_p95_1 = arrayfun(@(x) prctile(X_test.bmi(X_test.vital == 1 & X_test.age > x & X_test.age < x+d_age),95),age_ranges);
vital_bmi_age_p95_0 = arrayfun(@(x) prctile(X_test.bmi(X_test.vital == 0 & X_test.age > x & X_test.age < x+d_age),95),age_ranges);


h = figure;
h.Position(3:4) = [800 600];
centerfig(h);
hold all
plot(age_ranges + d_age/2,vital_bmi_age_m_1,'-ob')
plot(age_ranges + d_age/2,vital_bmi_age_p95_1,'--ob')
plot(age_ranges + d_age/2,vital_bmi_age_p5_1,'--ob')
plot(age_ranges + d_age/2,vital_bmi_age_m_0,'-or')
plot(age_ranges + d_age/2,vital_bmi_age_p95_0 + vital_bmi_age_s_0,'--or')
plot(age_ranges + d_age/2,vital_bmi_age_p5_0 - vital_bmi_age_s_0,'--or')


%% Mortality analysis - Logistic regression

k = 8;
var_r = {['SAI_' names{k}]};
idx_rv = cellfun(@(x) find(ismember(X_test.Properties.VariableNames, x)), {'vital'});
idx_pv = find(ismember(X_test.Properties.VariableNames,{'age','sex','bmi',var_r{1}}));
idx_data = ~any(ismissing(X_test(:,[idx_rv idx_pv])),2);

[logitCoef,dev,stats] = glmfit(X_test{idx_data, idx_pv},categorical(X_test{idx_data, idx_rv}),'binomial','logit');
logitFit = glmval(logitCoef,X_test{idx_data, idx_pv},'logit');
disp([stats.beta, stats.p]);

LL = stats.beta - 1.96.*stats.se;
UL = stats.beta + 1.96.*stats.se;

% mesh for grid
[age_grid,sai_grid] = meshgrid(age_vec,sai_vec);
logodds_grid = -logitCoef(1) - logitCoef(2)*age_grid - logitCoef(3)*sai_grid;
prob_grid = 1./(1 + exp(logodds_grid));
[min_prob,prob_line] = min(abs(prob_grid - 0.5));

h = figure;
h.Position(3:4) = [800 600];
centerfig(h);
im = imagesc(age_vec,sai_vec,prob_grid);
hold on
alpha_val = 0.2;
bap = scatter(X_test.age(X_test.vital == 1), X_test.(var_r{1})(X_test.vital == 1), 'b', 'filled');
bap.MarkerFaceAlpha = alpha_val;
bap.MarkerEdgeAlpha = alpha_val;
bap = scatter(X_test.age(X_test.vital == 0), X_test.(var_r{1})(X_test.vital == 0), 'r', 'filled');
bap.MarkerFaceAlpha = alpha_val;
bap.MarkerEdgeAlpha = alpha_val;

caxis([0 1])
colormap('bone')
cb = colorbar;
cb.Label.String = 'P(Alive)';
set(gca,'YDir','default')
xlabel('Age [Years]')
ylabel('SAI [Years]')
legend('Alive','Death')
set(gcf,'Color',[1 1 1]);
set( findall(h, '-property', 'fontsize'), 'fontsize', 10)
% export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\scatter_mae_' names{k}], '-pdf', '-transparent');
print('-dpdf',['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\shhs_vital_prelim_plot' names{k}])

%% Error in age ranges

age_ranges = 5:5:90;
n_age_range = nan(1,size(age_ranges,2)-1);
error_range = nan(3,size(age_ranges,2)-1);
for i = 1:size(age_ranges,2)-1
    if i == size(age_ranges,2)-1
        idx_age_range = X_test.age >= age_ranges(i) & X_test.age <= age_ranges(i+1);
    else
        idx_age_range = X_test.age >= age_ranges(i) & X_test.age < age_ranges(i+1);
    end
    n_age_range(i) = sum(idx_age_range);
    error_range(1,i) = prctile(X_test.SA(idx_age_range) - X_test.age(idx_age_range), 5);
    error_range(2,i) = mean(X_test.SA(idx_age_range) - X_test.age(idx_age_range));
    error_range(3,i) = prctile(X_test.SA(idx_age_range) - X_test.age(idx_age_range), 95);
    error_range(4,i) = mean(abs(X_test.SA(idx_age_range) - X_test.age(idx_age_range)));
end

h = figure;
h.Position(3:4) = [600 600];
centerfig(h);
hold all
plot(age_ranges(1:end-1) + mean(diff(age_ranges))/2, error_range(1,:),'--ok');
plot(age_ranges(1:end-1) + mean(diff(age_ranges))/2, error_range(2,:),'-ok');
plot(age_ranges(1:end-1) + mean(diff(age_ranges))/2, error_range(3,:),'--ok');
plot(age_ranges([1 end]),[0 0],'-r');
set(gca,'XTick',age_ranges);
grid on
ylim([-50 50])
xlabel('Chronological Age [years]')
ylabel('Brain Age - Chronological Age [years]')
legend('95th and 5th percentile','Mean')
set(gcf,'Color',[1 1 1]);
export_fig(gcf, 'C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\error_avg_per5_full_resp5', '-pdf', '-transparent');


error_range_avg = mean(abs(error_range(:,n_age_range > 10)),2);
disp(error_range_avg);

%% Error in age ranges (pre)
age_ranges = 5:5:90;
n_age_range = nan(1,size(age_ranges,2)-1);
error_range = nan(4,size(age_ranges,2)-1);
for i = 1:size(age_ranges,2)-1
    if i == size(age_ranges,2)-1
        idx_age_range = X_test.age >= age_ranges(i) & X_test.age <= age_ranges(i+1);
    else
        idx_age_range = X_test.age >= age_ranges(i) & X_test.age < age_ranges(i+1);
    end
    n_age_range(i) = sum(idx_age_range);
    error_range(1,i) = prctile(X_test.SA_pre(idx_age_range) - X_test.age(idx_age_range), 5);
    error_range(2,i) = mean(X_test.SA_pre(idx_age_range) - X_test.age(idx_age_range));
    error_range(3,i) = prctile(X_test.SA_pre(idx_age_range) - X_test.age(idx_age_range), 95);
    error_range(4,i) = mean(abs(X_test.SA_pre(idx_age_range) - X_test.age(idx_age_range)));
    
end

h = figure;
h.Position(3:4) = [600 600];
centerfig(h);
hold all
plot(age_ranges(1:end-1) + mean(diff(age_ranges))/2, error_range(1,:),'--ok');
plot(age_ranges(1:end-1) + mean(diff(age_ranges))/2, error_range(2,:),'-ok');
plot(age_ranges(1:end-1) + mean(diff(age_ranges))/2, error_range(3,:),'--ok');
plot(age_ranges([1 end]),[0 0],'-r');
set(gca,'XTick',age_ranges);
grid on
ylim([-50 50])
xlabel('Chronological Age [years]')
ylabel('Brain Age [years]')
set(gcf,'Color',[1 1 1]);
export_fig(gcf, 'C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\error_avg_per5_pre_resp5', '-pdf', '-transparent');


error_range_avg = mean(abs(error_range(:,n_age_range > 10)),2);
disp(error_range_avg);

%% Error plot
h = figure;
h.Position(3:4) = [600 600];
centerfig(h);
hold all
[N,c] = hist3([X_test.SA, X_test.age],'edges',{0:100, 0:100});
imagesc(0:100,0:100,N);
colormap gray
plot([0 100], [0 100],'--r','LineWidth',0.25)
grid minor
axis([10 90 10 90])
colorbar
xlabel('Chronological Age')
ylabel('Predicted Age')
set(gcf,'Color',[1 1 1]);
export_fig(gcf, 'C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\scatter_age_all', '-pdf', '-transparent');

%% Correlation plots for all
cohorts = {'CFS','STAGES','WSC','MrOS','SSC','SHHS'};
h = figure;
h.Position(3:4) = [600 600];
centerfig(h);
hold all
plot(X_test.age, X_test.SA,'.k');
plot([0 100], [0 100],'--r','LineWidth',0.25)
grid minor
xlabel('Chronological Age [years]')
ylabel('Brain Age [years]')
[r, p] = corrcoef(X_test.age, X_test.SA);
ax_min = min([floor(min(X_test.age)/5)*5 floor(min(X_test.SA)/5)*5]);
ax_max = max([ceil(max(X_test.age)/5)*5 ceil(max(X_test.SA)/5)*5]);
text(ax_min + 0.1*(ax_max - ax_min), ax_max - 0.1*(ax_max - ax_min), ['r = ' num2str(r(1,2))]);
axis([ax_min, ax_max, ax_min, ax_max]);

set(gcf,'Color',[1 1 1]);
% legend(horzcat(plot_cell{:}), cohorts, 'Location','SouthEast')
% export_fig(gcf, 'C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\corr_plot_all', '-pdf', '-transparent');

%% Error at various age ranges


%% Correlation plots for each cohort
cohorts = {'CFS','STAGES','WSC','MrOS','SSC','SHHS'};
h = figure;
h.Position(3:4) = [1200 600];
centerfig(h);
for i = 1:6
    subplot(2,3,i)
    hold all
    plot(X_test.age(X_test.cohort_code == i), X_test.SA(X_test.cohort_code == i),'.k');
    plot([0 100], [0 100],'--r','LineWidth',0.25)
    grid minor
    xlabel('Chronological Age [years]')
    ylabel('Brain Age [years]')
    title(cohorts{i});
    ax_min = min([floor(min(X_test.age(X_test.cohort_code == i))/5)*5 floor(min(X_test.SA(X_test.cohort_code == i))/5)*5]);
    ax_max = max([ceil(max(X_test.age(X_test.cohort_code == i))/5)*5 ceil(max(X_test.SA(X_test.cohort_code == i))/5)*5]);
    [r, p] = corrcoef(X_test.age(X_test.cohort_code == i), X_test.SA(X_test.cohort_code == i));
    text(ax_min + 0.1*(ax_max - ax_min), ax_max - 0.1*(ax_max - ax_min), ['r = ' num2str(r(1,2))]);
    axis([ax_min, ax_max, ax_min, ax_max]);
end
set(gcf,'Color',[1 1 1]);
export_fig(gcf, 'C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\corr_plot_all_cohorts', '-pdf', '-transparent');



%%
% Bland-altman plot for all cohorts
cohorts = {'CFS','STAGES','WSC','MrOS','SSC','SHHS'};
h = figure;
h.Position(3:4) = [1200 600];
centerfig(h);
for i = 1:6
    subplot(2,3,i)
    hold all
    x_bap = (X_test.age(X_test.cohort_code == i) + X_test.SA(X_test.cohort_code == i))/2;
    bap = scatter(x_bap,X_test.SAI(X_test.cohort_code == i),'filled');
    alpha_val = 0.5 + 0.45*(sum(X_test.cohort_code == 1)-length(x_bap))/sum(X_test.cohort_code == 6);
    bap.MarkerFaceAlpha = alpha_val;
    bap.MarkerEdgeAlpha = alpha_val;
    plot([0 100], [0 0],'--k','LineWidth',0.25)
    grid minor
    xlabel('(Sleep Age + Age)/2')
    ylabel('Sleep Age - Age')
    title(cohorts{i});
    axis([min(x_bap) max(x_bap) -50 50])
end
set(gcf,'Color',[1 1 1]);
export_fig(gcf, 'C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\bap_all_cohorts', '-pdf', '-transparent');



%% CPAP STAGES Analysis
%  Read apnea stats
STAGES_APNEA = readtable("H:\STAGES\PatientDemographicsAll.xlsx");

%  find all baseline in test
ids_stages = unique(cellfun(@(x) x(1:9), X_test.names(X_test.cohort_code == 2),'Un',0));
idx_cpap_bl = [];
idx_cpap_fu = [];
for i = 1:length(ids_stages)
    name_bl = [ids_stages{i} '_EDFAndScore'];
    idx_bl = ismember(X_test.names, name_bl);
    name_fu = [ids_stages{i} '_1_EDFAndScore'];
    idx_fu = ismember(X_test.names, name_fu);
    if any(idx_bl) && any(idx_fu) && X_test.cpap(idx_bl) == 0
        idx_cpap_bl = [idx_cpap_bl find(idx_bl)];
        idx_cpap_fu = [idx_cpap_fu find(idx_fu)];
    end
end

% Collect AHI, True Age and Sleep Age from subjects
AHI_bl = cellfun(@(x) STAGES_APNEA.ahi(ismember(STAGES_APNEA.s_code, x(1:end-12))), X_test.names(idx_cpap_bl));
AHI_fu = cellfun(@(x) STAGES_APNEA.ahi(ismember(STAGES_APNEA.s_code, x(1:end-12))), X_test.names(idx_cpap_fu));
Age_bl = X_test.age(idx_cpap_bl);
Age_fu = X_test.age(idx_cpap_fu);
% Sleep_Age
idx_SA_bl = cellfun(@(x) isfield(preds,([x '_hdf5'])), X_test.names(idx_cpap_bl));
idx_SA_fu = cellfun(@(x) isfield(preds,([x '_hdf5'])), X_test.names(idx_cpap_fu));
idx_SA = idx_SA_bl & idx_SA_fu;
SA_bl = cellfun(@(x) preds.([x '_hdf5']).age_p, X_test.names(idx_cpap_bl(idx_SA)));
SA_fu = cellfun(@(x) preds.([x '_hdf5']).age_p, X_test.names(idx_cpap_fu(idx_SA)));
% Sleep age index
SAI_bl = SA_bl - Age_bl(idx_SA);
SAI_fu = SA_fu - Age_fu(idx_SA);


h = figure;
h.Position(3:4) = [600 800];
centerfig(h);
plot(SAI_bl, AHI_bl(idx_SA),'o')

h = figure;
h.Position(3:4) = [600 800];
centerfig(h);
subplot(1,3,1)
plot([1 2],[SAI_bl SAI_fu]','ko-')
xlabel('Visit')
ylabel('SAI')
grid minor
subplot(1,3,2)
plot(-AHI_bl(idx_SA) + AHI_fu(idx_SA),-SAI_bl + SAI_fu,'ko');
xlabel('d(AHI)')
ylabel('d(SAI)')
grid minor
subplot(1,3,3)
plot([1 2],[AHI_bl(idx_SA) AHI_fu(idx_SA)]','ko-')
xlabel('Visit')
ylabel('AHI')
grid minor

%% Comparison to BAI model
bai_stages_path = 'C:\Users\andre\Dropbox\Phd\Stanford_extra\brain_age_eeg\stable_BA_STAGES.csv';
bai_stages = readtable(bai_stages_path);
bai_wsc_path = 'C:\Users\andre\Dropbox\Phd\Stanford_extra\brain_age_eeg\stable_BA_WSC.csv';
bai_wsc = readtable(bai_wsc_path);
bai_ssc_path = 'C:\Users\andre\Dropbox\Phd\Stanford_extra\brain_age_eeg\stable_BA_ApoE.csv';
bai_ssc = readtable(bai_ssc_path);

BAI = nan(size(X_test,1), 1);

stages_names = cellfun(@(x) x(1:max(1, length(x) - 12)), X_test.names, 'Un', 0);
idx_bai_stages = arrayfun(@(x) find(strcmp(stages_names, x), 1, 'first'), bai_stages.SID, 'Un', 0);
bai_stages.BAI(cellfun(@isempty, idx_bai_stages)) = nan;
idx_bai_stages(cellfun(@isempty, idx_bai_stages)) = {find(cellfun(@isempty, idx_bai_stages), 1, 'first')};
BAI(cell2mat(idx_bai_stages)) = bai_stages.BAI;

wsc_names = cellfun(@(x) x(1:max(1, length(x) - 7)), bai_wsc.SID, 'Un', 0);
idx_bai_wsc = arrayfun(@(x) find(strcmp(X_test.names, x), 1, 'first'), wsc_names, 'Un', 0);
bai_wsc.BAI(cellfun(@isempty, idx_bai_wsc)) = nan;
idx_bai_wsc(cellfun(@isempty, idx_bai_wsc)) = {find(cellfun(@isempty, idx_bai_wsc), 1, 'first')};
BAI(cell2mat(idx_bai_wsc)) = bai_wsc.BAI;

idx_bai_ssc = arrayfun(@(x) find(strcmp(X_test.names, x), 1, 'first'), bai_ssc.SID, 'Un', 0);
bai_ssc.BAI(cellfun(@isempty, idx_bai_ssc)) = nan;
idx_bai_ssc(cellfun(@isempty, idx_bai_ssc)) = {find(cellfun(@isempty, idx_bai_ssc), 1, 'first')};
BAI(cell2mat(idx_bai_ssc)) = bai_ssc.BAI;

X_test.BAI = BAI;

%%
in_c = [2 3 5];
idx_data = ismember(X_test.cohort_code, in_c);
h = figure;
h.Position(3:4) = [600 600];
centerfig(h);
plot(X_test.BAI(idx_data), X_test.SAI_ssc(idx_data), '.');
% scatter(X_test.BAI(idx_data), X_test.SAI_comb_avg(idx_data),[],X_test.remp(idx_data));
% colorbar
% caxis([0 50])
hold on
plot(get(gca,'XLim'), [0 0], '--k')
plot([0 0], get(gca,'YLim'), '--k')
xlabel('BAI [years]')
ylabel({'AEE [years]','(e, Ensemble - Avg.)'})
grid minor

[r,p] = corr(X_test.BAI(idx_data), X_test.SAI_comb_avg(idx_data), 'Type', 'Spearman', 'Rows', 'pairwise');
text(-40, 20, sprintf('r = %.2f, p = %.2g', r, p), 'HorizontalAlignment', 'center');
set(gcf,'Color',[1 1 1]);
set( findall(h, '-property', 'fontsize'), 'fontsize', 10);
export_fig(gcf, ['C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\figures\scatter_aee_bai'], '-pdf', '-transparent');


%% Age distribution of datasets
clear all; close all;
addpath(genpath('export_fig'));

%% Read data
%  SSC
ds_ssc_path = "H:\SSC\ssc.xlsx";
ds_ssc = readtable(ds_ssc_path,'Sheet','ssc');
ds_ssc(ds_ssc.age_float < 2,:) = [];
age_ssc = ds_ssc.age_float;
sex_ssc =  cellfun(@(x) strcmp(x, 'M'), ds_ssc.gender);
names_ssc = arrayfun(@(x) ['SSC_' num2str(x) '_1'], ds_ssc.patid,'Un',0);
plmi_ssc = ds_ssc.plm_index;
ahi_ssc = ds_ssc.apnea_hypopnea_index;

%  STAGES
ds_stages_path = 'H:\\STAGES\\PatientDemographics.xlsx';
ds_names_path = "H:\STAGES\filenames_all.txt";
ds_stages_bad1_path = "H:\STAGES\Discarded-Patients.xlsx";
ds_stages_bad2_path = "H:\STAGES\Bad_EDFs.xlsx";
ds_stages_ahi_path = "H:\STAGES\PatientDemographicsAll.xlsx";
bad_stages_1 = readtable(ds_stages_bad1_path);
bad_names_1 = bad_stages_1{~contains(bad_stages_1{:,2}, 'CPAP during recording'),1};
bad_stages_2 = readtable(ds_stages_bad2_path);
bad_names_2 = bad_stages_2{:,1};
names_stages =  readtable(ds_names_path,'ReadVariableNames',false,'Delimiter',',');
ds_stages_ahi = readtable(ds_stages_ahi_path);
names_stages = names_stages{:,1};
names_stages_short = names_stages;
age_stages = zeros(size(names_stages));
sex_stages = zeros(size(names_stages));
plmi_stages = zeros(size(names_stages));
ahi_stages = zeros(size(names_stages));
to_remove_stages = false(size(age_stages));
for i = 1:length(names_stages)
    idx = regexp(names_stages{i},'_EDFAndScore.zip','start');
    if isempty(idx)
        to_remove_stages(i) = true;
    else
        names_stages_short{i} = names_stages{i}(1:idx(1)-1);
        if any(strcmp(names_stages_short{i}, bad_names_1)) || any(strcmp(names_stages_short{i}, bad_names_2))
            to_remove_stages(i) = true;
        end
    end
end
ds_stages = readtable(ds_stages_path);
for i = 1:length(names_stages)
    idx = find(strcmp(ds_stages.s_code, names_stages_short{i}(1:9)));
    idx_full = find(strcmp(ds_stages_ahi.s_code, names_stages_short{i}));
    if isempty(idx) || isempty(idx_full)
        to_remove_stages(i) = true;
        continue;
    end
    age_stages(i) = ds_stages.age(idx(1));
    sex_stages(i) = strcmp('M', ds_stages.sex(idx(1)));
    ahi_stages(i) = ds_stages_ahi.ahi(idx_full(1));
    if age_stages(i) < 2 || isnan(age_stages(i)) || isempty(ds_stages.sex{idx(1)})
        to_remove_stages(i) = true;
    end
end
age_stages(to_remove_stages) = [];
sex_stages(to_remove_stages) = [];
names_stages(to_remove_stages) = [];
ahi_stages(to_remove_stages) = [];
plmi_stages(to_remove_stages) = [];
names_stages = cellfun(@(x) x(1:end-4), names_stages, 'Un', 0);
names_stages_short(to_remove_stages) = [];
% CFS
ds_cfs_path = 'G:\\cfs\\datasets\\cfs-visit5-dataset-0.4.0.csv';
ds_cfs = readtable(ds_cfs_path);
names_cfs = arrayfun(@(x) ['cfs-visit5-' num2str(x)],ds_cfs.nsrrid,'Un',0);
age_cfs = ds_cfs.age;
sex_cfs = ds_cfs.SEX;
ahi_cfs = ds_cfs.ahi_a0h3;
plmi_cfs = ds_cfs.AVGPLM;
% MrOS
ds_mros_path = 'G:\\mros\\datasets\\mros-visit1-dataset-0.3.0.csv';
ds_mros = readtable(ds_mros_path);
names_mros = cellfun(@(x) ['mros-visit1-' x], ds_mros.nsrrid,'Un',0);
age_mros = ds_mros.vsage1;
sex_mros = ones(size(age_mros));
ahi_mros = cellfun(@(x) str2num(x), ds_mros.pordi3pa, 'Un', 0);
ahi_mros(cellfun(@(x) isempty(x), ahi_mros)) = {NaN};
ahi_mros = cell2mat(ahi_mros);
plmi_mros = ds_mros.poavgplm;
% WSC
ds_wsc_path = 'G:\\WSC_PLM_ data_all.xlsx';
ds_wsc = readtable(ds_wsc_path);
ds_wsc = ds_wsc(~isnan(ds_wsc.age),:);
names_wsc = arrayfun(@(x) char([ds_wsc.SUBJ_ID{x} '_' num2str(ds_wsc.VISIT_NUMBER(x))]), 1:size(ds_wsc,1),'Un',0)';
age_wsc = ds_wsc.age;
sex_wsc = cellfun(@(x) strcmp(x, 'M'), ds_wsc.SEX);
ahi_wsc = ds_wsc.AHI4_ADJUSTED_V2;
plmi_wsc = ds_wsc.plmi_d140_s;
% SHHS
ds_shhs_path = 'H:\\shhs\\datasets\\shhs1-dataset-0.15.0.csv';
ds_shhs = readtable(ds_shhs_path);
age_shhs = ds_shhs.age_s1;
names_shhs = arrayfun(@(x) ['shhs1-' num2str(x)],ds_shhs.nsrrid,'Un',0);
sex_shhs = 2 - ds_shhs.gender;
ahi_shhs = ds_shhs.ahi_a0h3a;
plmi_shhs = nan(size(ahi_shhs));
% SOF
ds_sof_path = 'H:\sof\datasets\sof-visit-8-dataset-0.6.0.csv';
ds_sof = readtable(ds_sof_path);
ds_sof = ds_sof(~isnan(ds_sof.V8AGE),:);
age_sof = ds_sof.V8AGE;
names_sof = arrayfun(@(x) ['sof-visit-8-' num2str(x,'%05.f')],ds_sof.sofid,'Un',0);
sex_sof = zeros(size(age_sof));
ahi_sof = ds_sof.ahi_a0h3;
plmi_sof = ds_sof.noplm ./ (ds_sof.slpprdp / 60);
% HomePAP
ds_hpap_path = "H:\homepap\datasets\homepap-baseline-dataset-0.1.0.csv";
ds_hpap = readtable(ds_hpap_path);
ds_hpap = ds_hpap(~isnan(ds_hpap.age),:);
age_hpap = ds_hpap.age;
names_hpap = arrayfun(@(x) ['homepap-lab-full-' num2str(x)],ds_hpap.nsrrid,'Un',0);
sex_hpap = ds_hpap.gender;
ahi_hpap = ds_hpap.ahi;
plmi_hpap = nan(size(age_hpap));

%% Collect to table
T = table([age_cfs; age_stages; age_wsc; age_mros; age_ssc; age_shhs; age_sof; age_hpap], ...
    [repmat({'cfs'},length(age_cfs),1); repmat({'stages'},length(age_stages),1); repmat({'wsc'},length(age_wsc),1); repmat({'mros'},length(age_mros),1); repmat({'ssc'},length(age_ssc),1); repmat({'shhs'},length(age_shhs),1); repmat({'sof'},length(age_sof),1); repmat({'hpap'},length(age_hpap),1)],...
    [repmat(1,length(age_cfs),1); repmat(2,length(age_stages),1); repmat(3,length(age_wsc),1); repmat(4,length(age_mros),1); repmat(5,length(age_ssc),1); repmat(6,length(age_shhs),1); repmat(7,length(age_sof),1); repmat(8,length(age_hpap),1)],...
    (1:length([age_cfs; age_stages; age_wsc; age_mros; age_ssc; age_shhs; age_sof; age_hpap]))',...
    [names_cfs; names_stages; names_wsc; names_mros; names_ssc; names_shhs; names_sof; names_hpap], ...
    [sex_cfs; sex_stages; sex_wsc; sex_mros; sex_ssc; sex_shhs; sex_sof; sex_hpap],...
    [ahi_cfs; ahi_stages; ahi_wsc; ahi_mros; ahi_ssc; ahi_shhs; ahi_sof; ahi_hpap],...
    [plmi_cfs; plmi_stages; plmi_wsc; plmi_mros; plmi_ssc; plmi_shhs; plmi_sof; plmi_hpap],...
    'VariableNames', {'age', 'cohort', 'cohort_code', 'n', 'names','sex','ahi','plmi'});

%% Exclusion criteria
%  NO CPAP or BiPAP in train/val
%  NO RBD, PD or Narcolepsy in train/val
idx_cpap = false(size(T.cohort_code));
idx_narc = false(size(T.cohort_code));
idx_nd = false(size(T.cohort_code));
% SSC
bad_string_nd = {'REMBD','RBD','REM behavioral disorder'};
bad_string_narc = { 'Narcolepsy','Cataplexy','NARC'};
idx_nd_ssc = contains(ds_ssc.dx_1_other_string, bad_string_nd) | contains(ds_ssc.dx_1_string, bad_string_nd) | contains(ds_ssc.dx_2_string, bad_string_nd) | contains(ds_ssc.dx2_string, bad_string_nd);
idx_narc_ssc = contains(ds_ssc.dx_1_other_string, bad_string_narc) | contains(ds_ssc.dx_1_string, bad_string_narc) | contains(ds_ssc.dx_2_string, bad_string_narc) | contains(ds_ssc.dx2_string, bad_string_narc);
idx_T_ssc = find(T.cohort_code == 5);
idx_nd(idx_T_ssc) = idx_nd_ssc;
idx_narc(idx_T_ssc) = idx_narc_ssc;
% STAGES
ds_stages_cpap_path = "H:\STAGES\CPAP.xlsx";
ds_stages_bad1_path = "H:\STAGES\Discarded-Patients.xlsx";
cpap_stages = readtable(ds_stages_cpap_path);
cpap_stages = cpap_stages.Subjects;
bad_stages_1 = readtable(ds_stages_bad1_path);
cpap_names_1 = bad_stages_1{contains(bad_stages_1{:,2}, 'CPAP during recording'),1};
idx_T_stages = find(T.cohort_code == 2);
for i = 1:length(idx_T_stages)
    if length(T.names{idx_T_stages(i)}) > 21 || any(strcmp(names_stages_short{i},cpap_names_1)) || any(strcmp(names_stages_short{i},cpap_stages))
        idx_cpap(idx_T_stages(i)) = true;
    end
end
% CFS
idx_T_cfs = find(T.cohort_code == 1);
idx_cpap(idx_T_cfs) = ds_cfs.PRELCPAP;
idx_nd(idx_T_cfs) = (ds_cfs.PARKDIAG == 1);
idx_narc(idx_T_cfs) = (ds_cfs.DIANARC == 1);
% WSC
idx_T_wsc = find(T.cohort_code == 3);
idx_cpap(idx_T_wsc) = ~isnan(ds_wsc.cpap);
% MrOS
idx_T_mros = find(T.cohort_code == 4);
idx_narc(idx_T_mros) = (cellfun(@(x) strcmp(x, '1'), ds_mros.slnarc));
idx_nd(idx_T_mros) = (cellfun(@(x) strcmp(x, '1'), ds_mros.mhpark));
% SHHS
% no information

% SOF

% add to table
T.cpap = idx_cpap;
T.nd = idx_nd;
T.narc = idx_narc;


%  Exclude PSGs with missing data
h5_succes = readtable("H:\nAge\all_filenames.xlsx",'ReadVariableNames',false);
h5_succes = cellfun(@(x) x(1:end-5), h5_succes{:,1},'Un',0);
h5_succes(cellfun(@(x) length(x), h5_succes) == 14) = cellfun(@(x) x(1:7), h5_succes(cellfun(@(x) length(x), h5_succes) == 14), 'Un',0);
h5_succes_test_extra = dir(['H:\nAge\test_ext\*.hdf5']);
h5_succes_test_extra = cellfun(@(x) x(1:end-5), {h5_succes_test_extra.name},'Un',0);
T(~contains(T.names, [h5_succes; h5_succes_test_extra'],'IgnoreCase',true),:) = [];

%% Sample stradegy
%  Define dataset X_train
%  Divide in to N bins
%  while size(X) > train_size do
%    top_bin = bin with most samples (if tie choose randomly)
%    delete sample from top bin from most cohort best represented in top_bin 
%
%  Define dataset X_val (deleted samples)
%  Divide in to N bins
%  while size(Xd) > val_size do
%    top_bin = bin with most samples (if tie choose randomly)
%    delete random sample from top bin
%
%  Define data X_test (delted samples)
rng(260794);
N = 101;
NC = max(T.cohort_code);
edges = linspace(0,100,N);
X_train = T(T.cpap == 0 & T.nd == 0 & T.narc == 0 & T.cohort_code ~= 7 & T.cohort_code ~= 8,:);
train_size = 2500;
val_size = 200;
while size(X_train,1) > train_size
    [M,~] = histcounts(X_train.age, edges);
    top_bins = find(M == max(M));
    if length(top_bins) > 1
        top_bin = randsample(top_bins, 1);
    else
        top_bin = top_bins;
    end
    x_s = find(X_train.age >= edges(top_bin) & X_train.age < edges(top_bin + 1));
    if length(x_s) > 1
        n_cohorts = arrayfun(@(x) sum(X_train.cohort_code(x_s) == x), 1:NC);
        x_cohorts = find(n_cohorts == max(n_cohorts));
        if length(x_cohorts) > 1
            x_cohort = randsample(x_cohorts,1);
        else
            x_cohort = x_cohorts;
        end
        x_dt = x_s(X_train.cohort_code(x_s) == x_cohort);
        if length(x_dt) > 1
            %x_d = randsample(x_dt, 1);
            % find majority sex
            maj_sex = mean(X_train.sex(x_dt)) > 1/2;
            x_dt_s = x_dt(X_train.sex(x_dt) == maj_sex);
            if length(x_dt_s) > 1
                x_d = randsample(x_dt_s, 1);
            else
                x_d = x_dt_s;
            end
        else
            x_d = x_dt;
        end
    else
        x_d = x_s;
    end
    X_train(x_d,:) = [];
end

X_val = T(T.cpap == 0 & T.nd == 0 & T.narc == 0 & T.cohort_code ~= 7 & T.cohort_code ~= 8,:);
X_val(ismember(X_val.n, X_train.n),:) = [];
while size(X_val,1) > val_size
    [M,~] = histcounts(X_val.age, edges);
    top_bins = find(M == max(M));
    if length(top_bins) > 1
        top_bin = randsample(top_bins, 1);
    else
        top_bin = top_bins;
    end
    x_s = find(X_val.age >= edges(top_bin) & X_val.age < edges(top_bin + 1));
    if length(x_s) > 1
        n_cohorts = arrayfun(@(x) sum(X_val.cohort_code(x_s) == x), 1:NC);
        x_cohorts = find(n_cohorts == max(n_cohorts));
        if length(x_cohorts) > 1
            x_cohort = randsample(x_cohorts,1);
        else
            x_cohort = x_cohorts;
        end
        x_dt = x_s(X_val.cohort_code(x_s) == x_cohort);
        if length(x_dt) > 1
%             x_d = randsample(x_dt, 1);
            % find majority sex
            maj_sex = mean(X_val.sex(x_dt)) > 1/2;
            x_dt_s = x_dt(X_val.sex(x_dt) == maj_sex);
            if length(x_dt_s) > 1
                x_d = randsample(x_dt_s, 1);
            else
                x_d = x_dt_s;
            end
        else
            x_d = x_dt;
        end
    else
        x_d = x_s;
    end
    X_val(x_d,:) = [];
end

X_test = T;
X_test(ismember(X_test.n, [X_train.n; X_val.n]),:) = [];

%% Exclude all >= 90 years ???
X_test(X_test.age >= 90,:) = [];
X_test(X_test.age < 20,:) = [];

% Exclude SOF
X_test(X_test.cohort_code == 7,:) = [];

%% Visualize resulting distributions
%  Train
n_stages_train = histcounts(X_train.age(X_train.cohort_code == 2), edges);
n_cfs_train = histcounts(X_train.age(X_train.cohort_code == 1), edges);
n_mros_train = histcounts(X_train.age(X_train.cohort_code == 4), edges);
n_ssc_train = histcounts(X_train.age(X_train.cohort_code == 5), edges);
n_wsc_train = histcounts(X_train.age(X_train.cohort_code == 3), edges);
n_shhs_train = histcounts(X_train.age(X_train.cohort_code == 6), edges);
p_sex_train = arrayfun(@(x) mean(X_train.sex(edges(x) <= X_train.age & X_train.age < edges(x+1))), 1:length(edges)-1);
%  Val
n_stages_val = histcounts(X_val.age(X_val.cohort_code == 2), edges);
n_cfs_val = histcounts(X_val.age(X_val.cohort_code == 1), edges);
n_mros_val = histcounts(X_val.age(X_val.cohort_code == 4), edges);
n_ssc_val = histcounts(X_val.age(X_val.cohort_code == 5), edges);
n_wsc_val = histcounts(X_val.age(X_val.cohort_code == 3), edges);
n_shhs_val = histcounts(X_val.age(X_val.cohort_code == 6), edges);
p_sex_val = arrayfun(@(x) mean(X_val.sex(edges(x) <= X_val.age & X_val.age < edges(x+1))), 1:length(edges)-1);
%  Test
n_stages_test = histcounts(X_test.age(X_test.cohort_code == 2), edges);
n_cfs_test = histcounts(X_test.age(X_test.cohort_code == 1), edges);
n_mros_test = histcounts(X_test.age(X_test.cohort_code == 4), edges);
n_ssc_test = histcounts(X_test.age(X_test.cohort_code == 5), edges);
n_wsc_test = histcounts(X_test.age(X_test.cohort_code == 3), edges);
n_shhs_test = histcounts(X_test.age(X_test.cohort_code == 6), edges);
% n_sof_test = histcounts(X_test.age(X_test.cohort_code == 7), edges);
n_hpap_test = histcounts(X_test.age(X_test.cohort_code == 8), edges);
p_sex_test = arrayfun(@(x) mean(X_test.sex(edges(x) <= X_test.age & X_test.age < edges(x+1))), 1:length(edges)-1);

h = figure;
h.Position(3:4) = [800 600];
centerfig(h);
subplot(3,1,1)
bar(edges(1:end-1), [n_stages_test; n_cfs_test; n_wsc_test; n_mros_test; n_ssc_test; n_shhs_test; n_hpap_test],1,'stacked')
ylabel('Count');
legend({'STAGES','CFS','WSC','MrOS','SSC','SHHS','HomePAP'},'Location','NorthWest')
set(gca,'XtickLabel',{});
axis([0 95 0 4*size(X_test,1)/100])
title('Test Set')
grid on
% subplot(9,1,3)
% plot(edges(1:end-1), 1-p_sex_test, '-.', 'Color', [0.5 0.5 0.5]);
% axis([0 100 -0.05 1.05])
% set(gca,'YTick',0:0.25:1);
% set(gca,'XtickLabel',{});
% ylabel('% Female')
subplot(3,1,2)
bar(edges(1:end-1), [n_stages_val; n_cfs_val; n_wsc_val; n_mros_val; n_ssc_val; n_shhs_val],1,'stacked')
ylabel('Count');
title('Validation Set')
axis([0 95 0 2*val_size/100])
set(gca,'XtickLabel',{});
grid on
% subplot(9,1,6)
% plot(edges(1:end-1), 1-p_sex_val, '-.', 'Color', [0.5 0.5 0.5]);
% axis([0 100 -0.05 1.05])
% set(gca,'YTick',0:0.25:1);
% ylabel('% Female')
% set(gca,'XtickLabel',{});
subplot(3,1,3)
bar(edges(1:end-1), [n_stages_train; n_cfs_train; n_wsc_train; n_mros_train; n_ssc_train; n_shhs_train],1,'stacked')
ylabel('Count');
title('Training Set')
axis([0 95 0 1.5*train_size/100])
% set(gca,'XtickLabel',{});
% subplot(9,1,9)
% plot(edges(1:end-1), 1-p_sex_train, '-.', 'Color', [0.5 0.5 0.5]);
% axis([0 100 -0.05 1.05])
% set(gca,'YTick',0:0.25:1);
% ylabel('% Female')
xlabel('Age [years]')
grid on
set(gcf,'Color',[1 1 1]);
export_fig(gcf, 'dataset_age_distribution', '-pdf', '-transparent');

fprintf('\n');
fprintf('\tCFS\tSTAGES\tWSC\tMrOS\tSSC\tSHHS\tSOF\tHomePAP\tAll\n');
fprintf('Train (n)\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n',sum(n_cfs_train),sum(n_stages_train),sum(n_wsc_train),sum(n_mros_train),sum(n_ssc_train),sum(n_shhs_train),0,0,size(X_train,1));
fprintf('Age (\x03bc \x00B1 \x03c3)\t');
for i = 1:NC
    fprintf('%.2f \x00B1 %.2f\t',mean(X_train.age(X_train.cohort_code == i)),std(X_train.age(X_train.cohort_code == i)));
end
fprintf('%.2f \x00B1 %.2f\t\n',mean(X_train.age),std(X_train.age));
fprintf('Val (n)\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n',sum(n_cfs_val),sum(n_stages_val),sum(n_wsc_val),sum(n_mros_val),sum(n_ssc_val),sum(n_shhs_val),0,0,size(X_val,1));
fprintf('Age (\x03bc \x00B1 \x03c3)\t');
for i = 1:NC
    fprintf('%.2f \x00B1 %.2f\t',mean(X_val.age(X_val.cohort_code == i)),std(X_val.age(X_val.cohort_code == i)));
end
fprintf('%.2f \x00B1 %.2f\t\n',mean(X_val.age),std(X_val.age));
fprintf('Test (n)\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n',sum(n_cfs_test),sum(n_stages_test),sum(n_wsc_test),sum(n_mros_test),sum(n_ssc_test),sum(n_shhs_test),0,sum(n_hpap_test),size(X_test,1));
fprintf('Age (\x03bc \x00B1 \x03c3)\t');
for i = 1:NC
    fprintf('%.2f \x00B1 %.2f\t',mean(X_test.age(X_test.cohort_code == i)),std(X_test.age(X_test.cohort_code == i)));
end
fprintf('%.2f \x00B1 %.2f\t\n',mean(X_test.age),std(X_test.age));
fprintf('\n');

%% Print to list
writetable(X_train,'H:\nAge\X_train.csv');
writetable(X_val,'H:\nAge\X_val.csv');
writetable(X_test,'H:\nAge\X_test.csv');

%% Print plmi - age association
% plmi_cohort = [1 3 4 5];
% plmi_cohort_names = {'CFS','WSC','MrOS','SSC','All'};
% col=@(x)reshape(x,numel(x),1);
% violinplot2=@(C,varargin)violinplot(cell2mat(cellfun(col,col(C),'uni',0)),cell2mat(arrayfun(@(I)I*ones(numel(C{I}),1),col(1:numel(C)),'uni',0)),varargin{:});
% 
% h = figure;
% h.Position(3:4) = [600 600];
% centerfig(h);
% hold all
% for i = plmi_cohort
%     plot(T.age(T.cohort_code == i), (T.plmi(T.cohort_code == i)).^(1/3),'.')
% end
% xlim([0 100])
% legend(plmi_cohort_names,'Location','NorthWest')
% xlabel('Age [years]')
% ylabel('PLMI [h^{-1}]')
% set(gca,'YTickLabel',get(gca,'YTick').^3);
% grid minor
% % set(gcf,'Color',[1 1 1]);
% % export_fig(gcf, 'plmi_age_plot', '-pdf', '-transparent');
% 
% 
% h = figure;
% h.Position(3:4) = [600 600];
% centerfig(h);
% fprintf('\n');
% fprintf('\tCFS\tWSC\tMrOS\tSSC\tAll\n');
% da = 10;
% plmi_split_all = cell(9,1);
% plmi_data_all = cell(9,length(plmi_cohort)+1);
% for ar = 0:da:(90 - da)
%     fprintf('Age [%d - %d]\t',ar,ar+da);
%     hold all
%     plmi_split = zeros(length(plmi_cohort)+1,3);
%     for i = 1:length(plmi_cohort)
%         plmi_data = T.plmi(T.age > ar & T.age <= ar+da & T.cohort_code == plmi_cohort(i) & ~isnan(T.plmi));
%         plmi_data_all{(ar+da)/da,i} = plmi_data;
%         plmi_split(i,:) = [mean(plmi_data == 0) mean(plmi_data < 15 & plmi_data > 0) mean(plmi_data >= 15)];
%         fprintf('%.2f \x00B1 %.2f (n = %.0f)\t',mean(plmi_data), std(plmi_data), length(plmi_data));
%     end
%     plmi_data = T.plmi(T.age > ar & T.age <= ar+da & ismember(T.cohort_code, plmi_cohort) & ~isnan(T.plmi));
%     plmi_data_all{(ar+da)/da,end} = plmi_data;
%     plmi_split(end,:) = [mean(plmi_data == 0) mean(plmi_data < 15 & plmi_data > 0) mean(plmi_data >= 15)];
%     plmi_split_all{(ar+da)/da} = plmi_split;
%     fprintf('%.2f \x00B1 %.2f (n = %.0f)\t',mean(plmi_data), std(plmi_data), length(plmi_data));
% %     fprintf('%.0f\t',length(plmi_data));
%     fprintf('\n');
% end
% 
% for i = 1:length(plmi_cohort)+1
%     subplot(length(plmi_cohort)+1,1,i)
%     bar(cell2mat(cellfun(@(x) x(i,:), plmi_split_all,'Un',0)),'stacked')
%     set(gca,'XTick',1:9)
%     set(gca,'XTickLabel',arrayfun(@(x) ['[' num2str(x) ' - ' num2str(x+10) ']'],0:da:(90-da),'Un',0))
%     if i == 3
%         legend({'PLMI = 0','0 < PLMI < 15', 'PLMI >= 15'},'Location','West');
%     end
%     ylabel(plmi_cohort_names{i});
% end
% xlabel('Age range [years]')
% set(gcf,'Color',[1 1 1]);
% export_fig(gcf, 'plmi_age_bar_plot', '-pdf', '-transparent');
% 
% h = figure;
% h.Position(3:4) = [600 600];
% centerfig(h);
% for i = 1:length(plmi_cohort)+1
%     subplot(length(plmi_cohort)+1,1,i)
%     violinplot2(cellfun(@(x) x.^(1/3), plmi_data_all(:,i),'Un',0));
%     ylabel('PLMI [h^{-1}]')
%     set(gca,'YTickLabel',get(gca,'YTick').^3);
%     set(gca,'XTick',1:9)
%     set(gca,'XTickLabel',arrayfun(@(x) ['[' num2str(x) ' - ' num2str(x+10) ']'],0:da:(90-da),'Un',0))
%     if i == 3
%         legend({'PLMI = 0','0 < PLMI < 15', 'PLMI >= 15'},'Location','West');
%     end
%     ylabel(plmi_cohort_names{i});
% end
% xlabel('Age range [years]')
% % set(gcf,'Color',[1 1 1]);
% % export_fig(gcf, 'plmi_age_violin_plot', '-pdf', '-transparent');
% 
% 
% 
% T_lm = T(ismember(T.cohort_code, plmi_cohort),:);
% T_lm.cohort_dummy = categorical(T_lm.cohort);
% idx_pv = find(ismember(T_lm.Properties.VariableNames,{'age','cohort_dummy'}));
% idx_rv = find(ismember(T_lm.Properties.VariableNames,{'plmi'}));
% idx_data = ismember(T_lm.cohort_code, plmi_cohort) & ~any(ismissing(T_lm(:,[idx_rv idx_pv])),2);
% mdl = fitlm(T_lm(idx_data,:),'ResponseVar',idx_rv,'PredictorVars',idx_pv);
% disp(mdl);
% 
% %% Export plmi gwas
% T_gwas = readtable("G:\cfs\datasets\cfs-nsrr-to-dbgap-ids-20201130_v2.txt");
% %T_gwas_have_psg = readtable("G:\cfs\datasets\CFS_GWAS_new.csv");
% T_gwas_d1 = readtable("G:\cfs\datasets\CFS_Affy_6_dbGaP.fam", 'FileType','text');
% T_gwas_d2 = readtable("G:\cfs\datasets\CFS_iSelect_dbGaP.fam", 'FileType','text');
% 
% gwas_link = cellfun(@(x) str2num(x(end-5:end)), T.names(T.cohort_code == 1));
% gwas_id = arrayfun(@(x) find(x == gwas_link), T_gwas.nsrr_obf_pptid);
% 
% T_gwas.plmi = T(T.cohort_code == 1,:).plmi(gwas_id);
% T_gwas.ahi = T(T.cohort_code == 1,:).ahi(gwas_id);
% T_gwas.age = T(T.cohort_code == 1,:).age(gwas_id);
% T_gwas.sex = T(T.cohort_code == 1,:).sex(gwas_id);
% T_gwas.bmi = ds_cfs.bmi(gwas_id);
% T_gwas.black = ds_cfs.BLACK(gwas_id);
% 
% gwas_link_2 = cellfun(@(x) find(strcmp(T_gwas.dbgap_local_subject_id, x)),T_gwas_d1.Var2,'Un',0);
% T_gwas_d1.has_psg = cellfun(@(x) ~isempty(x), gwas_link_2);
% gwas_link_3 = [gwas_link_2{T_gwas_d1.has_psg}];
% T_gwas_d1.nsrrid = nan(size(T_gwas_d1,1),1);
% T_gwas_d1.nsrrid(T_gwas_d1.has_psg) = T_gwas.nsrr_obf_pptid(gwas_link_3);
% T_gwas_d1.age_at_psg = nan(size(T_gwas_d1,1),1);
% T_gwas_d1.age_at_psg(T_gwas_d1.has_psg) = T_gwas.age(gwas_link_3);
% T_gwas_d1.bmi_at_psg = nan(size(T_gwas_d1,1),1);
% T_gwas_d1.bmi_at_psg(T_gwas_d1.has_psg) = T_gwas.bmi(gwas_link_3);
% T_gwas_d1.sex = nan(size(T_gwas_d1,1),1);
% T_gwas_d1.sex(T_gwas_d1.has_psg) = T_gwas.sex(gwas_link_3);
% T_gwas_d1.ahi = nan(size(T_gwas_d1,1),1);
% T_gwas_d1.ahi(T_gwas_d1.has_psg) = T_gwas.ahi(gwas_link_3);
% T_gwas_d1.plmi = nan(size(T_gwas_d1,1),1);
% T_gwas_d1.plmi(T_gwas_d1.has_psg) = T_gwas.plmi(gwas_link_3);
% T_gwas_d1.black = nan(size(T_gwas_d1,1),1);
% T_gwas_d1.black(T_gwas_d1.has_psg) = T_gwas.black(gwas_link_3);
% 
% gwas_link_2 = cellfun(@(x) find(strcmp(T_gwas.dbgap_local_subject_id, x)),T_gwas_d2.Var2,'Un',0);
% T_gwas_d2.has_psg = cellfun(@(x) ~isempty(x), gwas_link_2);
% gwas_link_3 = [gwas_link_2{T_gwas_d2.has_psg}];
% T_gwas_d2.nsrrid = nan(size(T_gwas_d2,1),1);
% T_gwas_d2.nsrrid(T_gwas_d2.has_psg) = T_gwas.nsrr_obf_pptid(gwas_link_3);
% T_gwas_d2.age_at_psg = nan(size(T_gwas_d2,1),1);
% T_gwas_d2.age_at_psg(T_gwas_d2.has_psg) = T_gwas.age(gwas_link_3);
% T_gwas_d2.bmi_at_psg = nan(size(T_gwas_d2,1),1);
% T_gwas_d2.bmi_at_psg(T_gwas_d2.has_psg) = T_gwas.bmi(gwas_link_3);
% T_gwas_d2.sex = nan(size(T_gwas_d2,1),1);
% T_gwas_d2.sex(T_gwas_d2.has_psg) = T_gwas.sex(gwas_link_3);
% T_gwas_d2.ahi = nan(size(T_gwas_d2,1),1);
% T_gwas_d2.ahi(T_gwas_d2.has_psg) = T_gwas.ahi(gwas_link_3);
% T_gwas_d2.plmi = nan(size(T_gwas_d2,1),1);
% T_gwas_d2.plmi(T_gwas_d2.has_psg) = T_gwas.plmi(gwas_link_3);
% T_gwas_d2.black = nan(size(T_gwas_d2,1),1);
% T_gwas_d2.black(T_gwas_d2.has_psg) = T_gwas.black(gwas_link_3);
% 
% % gwas_link_2 = cellfun(@(x) find(strcmp(T_gwas.dbgap_local_subject_id, x)),T_gwas_have_psg.PHENO_ID,'Un',0);
% % T_gwas_have_psg.has_psg = cellfun(@(x) ~isempty(x), gwas_link_2);
% % gwas_link_3 = [gwas_link_2{T_gwas_have_psg.has_psg}];
% % T_gwas_have_psg.nsrrid = nan(size(T_gwas_have_psg,1),1);
% % T_gwas_have_psg.nsrrid(T_gwas_have_psg.has_psg) = T_gwas.nsrr_obf_pptid(gwas_link_3);
% % T_gwas_have_psg.age_at_psg = nan(size(T_gwas_have_psg,1),1);
% % T_gwas_have_psg.age_at_psg(T_gwas_have_psg.has_psg) = T_gwas.age(gwas_link_3);
% % T_gwas_have_psg.bmi_at_psg = nan(size(T_gwas_have_psg,1),1);
% % T_gwas_have_psg.bmi_at_psg(T_gwas_have_psg.has_psg) = T_gwas.bmi(gwas_link_3);
% % T_gwas_have_psg.sex = nan(size(T_gwas_have_psg,1),1);
% % T_gwas_have_psg.sex(T_gwas_have_psg.has_psg) = T_gwas.sex(gwas_link_3);
% % T_gwas_have_psg.ahi = nan(size(T_gwas_have_psg,1),1);
% % T_gwas_have_psg.ahi(T_gwas_have_psg.has_psg) = T_gwas.ahi(gwas_link_3);
% % T_gwas_have_psg.plmi = nan(size(T_gwas_have_psg,1),1);
% % T_gwas_have_psg.plmi(T_gwas_have_psg.has_psg) = T_gwas.plmi(gwas_link_3);
% % T_gwas_have_psg.black = nan(size(T_gwas_have_psg,1),1);
% % T_gwas_have_psg.black(T_gwas_have_psg.has_psg) = T_gwas.black(gwas_link_3);
% 
% writetable(T_gwas_d1,'C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\matlab\CFS_Affy_6_dbGaP_plmi_gwas_linked.fam','FileType','text');
% writetable(T_gwas_d2,'C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\matlab\.fam','FileType','text');


function [preds, metrics, metrics_pre] = read_model_metrics(path, extra_name)
if ~exist('extra_name','var')
    extra_name = '';
end
fid_m  = fopen([path '\predictions' extra_name '.json']);
raw_m  = fread(fid_m, inf);
str_m  = char(raw_m);
fclose(fid_m);
preds = jsondecode(str_m');
% metrics
metrics = readtable([path '\metrics' extra_name '.csv']);
if exist([path '\metrics_pre' extra_name '.csv'],'file')
    metrics_pre = readtable([path '\metrics_pre' extra_name '.csv']);
else
    metrics_pre = metrics;
end
end
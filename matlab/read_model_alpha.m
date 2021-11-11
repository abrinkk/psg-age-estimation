function [preds, preds_pre] = read_model_alpha(path, record, only_pre, only_full)
if only_pre
    preds = [];
else
    fid_m  = fopen([path '\predictions.json']);
    raw_m  = fread(fid_m, inf);
    str_m  = char(raw_m);
    fclose(fid_m);
    preds = jsondecode(str_m');
end
if only_full
    preds_pre = [];
else
    % metrics
    fid_m  = fopen([path '\predictions_pre_sep\' record '.json']);
    raw_m  = fread(fid_m, inf);
    str_m  = char(raw_m);
    fclose(fid_m);
    preds_pre = jsondecode(str_m');
end
end
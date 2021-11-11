function [alpha, alpha_pre] = match_record_alpha(preds, preds_pre, record, cohort_code)

% Record
metrics_fields = fieldnames(preds);
metrics_fields7 = cellfun(@(x) x(1:7), metrics_fields,'Un',0);
id = strrep(record,'-','_');
id = [id '_hdf5'];
if cohort_code == 3
    id = metrics_fields{strcmp(metrics_fields7, id(1:end-5))};
elseif cohort_code == 4
    id = lower(id);
end
if isfield(preds, id)
    alpha = preds.(id).alpha;
end

if ~isempty(preds_pre)
    % Record (pre)
    metrics_fields = fieldnames(preds_pre);
    metrics_fields7 = cellfun(@(x) x(1:7), metrics_fields,'Un',0);
    id = strrep(record,'-','_');
    id = [id '_hdf5'];
    if cohort_code == 3
        id = metrics_fields{strcmp(metrics_fields7, id(1:end-5))};
    elseif cohort_code == 4
        id = lower(id);
    end
    if isfield(preds_pre, id)
        alpha_pre = preds_pre.(id).alpha;
    end
else
    alpha_pre = -1;
end
end



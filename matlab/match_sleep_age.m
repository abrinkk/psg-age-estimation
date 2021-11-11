function [SA, SA_pre] = match_sleep_age(preds, metrics_pre, X_test, rep_str)

% Replace strings
preds = cell2struct( struct2cell(preds), strrep(fieldnames(preds),'_EDFAndScore',''));
metrics_pre.record = strrep(metrics_pre.record,rep_str,'');

% Add sleep age and sleep age index
SA = nan(size(X_test,1),1);
metrics_fields = fieldnames(preds);
metrics_fields7 = cellfun(@(x) x(1:7), metrics_fields,'Un',0);
for i = 1:size(X_test,1)
    id = strrep(X_test.names{i},'-','_');
    id = strrep(id, rep_str, '');
    id = [id '_hdf5'];
    if X_test.cohort_code(i) == 3
        id = metrics_fields{strcmp(metrics_fields7, id(1:end-5))};
    elseif X_test.cohort_code(i) == 4
        id = lower(id);
    end
    if isfield(preds, id)
        SA(i) = preds.(id).age_p;
    end
end

% Add sleep age for pre-preds
SA_pre = nan(size(X_test,1),1);
metrics_fields = metrics_pre.record;
metrics_fields7 = cellfun(@(x) x(1:7), metrics_fields,'Un',0);
for i = 1:size(X_test, 1)
    id = X_test.names{i};
    id = strrep(id, rep_str, '');
    id = [id '.hdf5'];
    if X_test.cohort_code(i) == 3
        id = metrics_fields{strcmp(metrics_fields7, id(1:end-5))};
    elseif X_test.cohort_code(i) == 4
        id = lower(id);
    end
    match = find(strcmp(metrics_pre.record, id));
    if ~isempty(match)
        SA_pre(i) = metrics_pre.age_p(match);
    end
end
end
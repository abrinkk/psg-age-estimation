function rel_sum = summarize_rel_stat(rel_stat,g_win,sum_dim)
% Input is cell of size (N,1)
idx = cellfun(@(x) ~isempty(x), rel_stat);
rel_stat = rel_stat(idx);
if size(rel_stat,1) > 1
    rel_stat = rel_stat';
end
if isempty(rel_stat)
    rel_avg = 0;
    rel_std = 0;
else
    if size(rel_stat{1},2) > 1
        rel_stat_mat = cat(3, rel_stat{:});
    else
        rel_stat_mat = cell2mat(rel_stat);
    end
    rel_stat_filt = convn(rel_stat_mat,g_win,'same')/sum(g_win);
    rel_std = std(rel_stat_filt,[], sum_dim, 'omitnan');
    rel_avg = mean(rel_stat_filt, sum_dim, 'omitnan');
end
rel_sum = {rel_avg, rel_std, idx};
end
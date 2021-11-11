


figure
for j = 1:9
    r = nan(10000,1);
    
    for i = 1:length(r)
        t = randn(32,1)*20 + 60;
        p = randn(32,1)*5 + t*j/10 + (1-j/10)*60;
        r_temp = corrcoef(p - t, t);
        r(i) = abs(r_temp(1,2)).^4;
    end
    subplot(3,3,j)
    histogram(r);
    title(num2str(j/10))
    xlim([0 1])
end
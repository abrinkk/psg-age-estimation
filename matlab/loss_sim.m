epsilon = 10^(-10);

loss_huber = @(p, t) mean((abs(p-t) < 5).*(1/2*abs(p-t).^2) + (abs(p-t) >= 5).*(5*abs(p-t) - 1/2*5^2))/68.13;
loss_l1    = @(p, t) mean((5*abs(p-t)))/68.13;
loss_corr  = @(p, t) 1/2*((abs(sum(((p-t)-mean((p-t))).*(t-mean(t))) ./ (epsilon + sqrt(epsilon + sum(((p-t)-mean(((p-t)))).^2))*sqrt(epsilon + sum((t-mean(t)).^2))))).^2);
loss_cov   = @(p, t) 0.0025*abs((sum((-min(0,((p-t)-mean((p-t))).*(t-mean(t)))))))/(length(p)-1);
loss = @(p, t) loss_huber(p,t) + loss_cov(p,t);
loss_l1_cov = @(p, t) loss_l1(p,t) + loss_cov(p,t);

loss_huber_c = @(p, t) mean((abs(p-t) < 5).*(1/2*abs(p-t).^2).*((t > 60 & (t > p)) | (t < 60 & (t < p))) + (abs(p-t) >= 5).*(5*abs(p-t) - 1/2*5^2))/68.13;
loss_huber_f0 = @(p,t) (abs(p-t) > 2.5 & abs(p-t) <= 7.5).*(1/2*(abs(p-t)-2.5).^2) + (abs(p-t) > 7.5).*(5*(abs(p-t)-2.5)-1/2*5^2);
loss_huber_f = @(p, t) mean((t > 60).*(loss_huber_f0(p-2.5,t)) + (t <= 60).*(loss_huber_f0(p+2.5,t)))/68.13;

%% Plots

% t = [40 60];
t = [70, 40];
p_g = repmat(20:0.1:90,2,1);

[P_1,P_2] = meshgrid(p_g(1,:),p_g(2,:));
P = [reshape(P_1,[],1), reshape(P_2,[],1)];
L = arrayfun(@(x) loss_huber_f(P(x,:),t), 1:size(P,1));
l = reshape(L,size(P_1,1),size(P_2,2));

%%
figure
contour(p_g(1,:),p_g(2,:),l,epsilon+(0:0.5:10))
% imagesc(p_g(1,:),p_g(2,:),l)
set(gca,'YDir','default')
hold on
plot(t(1),t(2),'rx');
text(t(1)+1,t(2)-1,'y')
colorbar
xlabel('p_1')
ylabel('p_2')

%% Correalation and covariance simulations
% 
x = 0:0.01:10;
a = 1;
figure
for i = 1:9
    subplot(3,3,i)
    y = a*x + randn(size(x))*i;
    plot(x,y,'.');
    cov_xy = cov(x,y);
    cor_xy = corrcoef(x,y);
    text(1,9,num2str(cor_xy(1,2)));
    text(9,1,num2str(cov_xy(1,2)));
    xlim([0, 10]);
    ylim([-20, 30]);
    
end
    
    
%%

x1 = randn(511,1)*19.3 + 44.1;
x2 = randn(1022,1)*11.2 + 63.1;
x3 = randn(1022,1)*8.4 + 59.7;
x = [x1;x2;x3];

bs = 128;

x_batch = [randsample(x1,round(bs/5));randsample(x2,round(2*bs/5));randsample(x3,round(2*bs/5))];
y_batch = randn(bs,1)*5 + [x_batch(1:round(bs/5)); x_batch(1+round(bs/5):end)*0.9 + 0.1*mean(x)];

xy_cov = cov(y_batch-x_batch,x_batch);
xy_l1  = mean(abs(y_batch-x_batch));

figure
plot(x_batch,y_batch-x_batch,'.');
hold on
plot(mean(x)*ones(2,1),[-50 50],'--r')
plot([0 100],mean(y_batch-x_batch)*ones(2,1),'--r')
text(20,10,num2str(xy_cov(2,1)))
text(80,-10,num2str(xy_l1));
axis([0 100 -20 20])





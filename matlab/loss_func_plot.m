% Loss function plot

f_huber = @(y,t) ((5*abs(y-t) - 25/2).*(abs(y-t) > 5) + (1/2*abs(y-t).^2).*(abs(y-t) <= 5))/112.5;
f_l1 = @(y,t) (abs(y-t))/25;
f_l2 = @(y,t) (abs(y-t).^2)/(25^2);
f_normal = @(m,s,t) -log(pdf('Normal',t,m,s)) / 6.3465;
f_gamma = @(m,s,t) -log(pdf('Gamma',t,m/s,s)) / 5.0388;



t = 40;
y = 0:0.1:100;
s = 10;
h = figure;
h.Position(3:4) = [800 400];
centerfig(h);
hold all
p_huber = plot(y, f_huber(y,t));
p_l1 = plot(y, f_l1(y,t));
p_l2 = plot(y, f_l2(y,t));
p_normal = plot(y, f_normal(y,s,t));
plot([t t],[0 100],'--k');
ylabel('Loss')
xlabel('Predicted Age [years]')
legend([p_normal, p_huber, p_l1, p_l2],{'NLL Normal (\sigma = 10)','Huber loss','L1 loss','L2 loss'},'Location','North');
axis([0 90 0 5])
grid minor

t = 40;
y = 0:0.1:100;
s = 10;
h = figure;
h.Position(3:4) = [800 400];
centerfig(h);
hold all
p_huber = plot(y(2:end), -diff(f_huber(y,t))./diff(y));
p_l1 = plot(y(2:end), -diff(f_l1(y,t))./diff(y));
p_l2 = plot(y(2:end), -diff(f_l2(y,t))./diff(y));
p_normal = plot(y(2:end), -diff(f_normal(y,s,t))./diff(y));
plot([t t],[-100 100],'--k');
plot([0 100],[0 0],'--k');
ylabel({'-\delta(Loss) / \delta(y)'})
xlabel('Predicted Age [years]')
legend([p_normal, p_huber, p_l1, p_l2],{'NLL Normal (\sigma = 10)','Huber loss','L1 loss','L2 loss'},'Location','North');
axis([0 90 -0.2 0.2])
grid minor

t = 40;
y = 50;
s = 1:50;
h = figure;
h.Position(3:4) = [800 400];
centerfig(h);
hold all
p_normal_10 = plot(s(2:end), -diff(f_normal(50,s,t))./diff(s));
p_normal_20 = plot(s(2:end), -diff(f_normal(60,s,t))./diff(s));
p_normal_30 = plot(s(2:end), -diff(f_normal(70,s,t))./diff(s));
plot([1 50],[0 0],'--k');
ylabel({'-\delta(Loss) / \delta(\sigma)'})
xlabel('\sigma [years]')
legend([p_normal_10, p_normal_20, p_normal_30],{'NLL Normal (|y - t| = 10)','NLL Normal (|y - t| = 20)','NLL Normal (|y - t| = 30)'},'Location','North');
axis([1 50 -0.05 0.2])
grid minor


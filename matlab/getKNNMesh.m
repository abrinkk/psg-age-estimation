function [X, Y, Z, h_idx] = getKNNMesh(x, y, z, k, plot_opt)


% Meshgrid
n = 1000;
x_v = linspace(min(x), max(x), n);
y_v = linspace(min(y), max(y), n);
[X, Y] = meshgrid(x_v, y_v);

% KNN search
idx = knnsearch([x y], [X(:), Y(:)], 'K', k, 'Distance', 'seuclidean');
Z = mean(z(idx),2);
Z = reshape(Z, [], n);

% Convex hull
h_idx = convhull(x, y);

if plot_opt
    h = figure;
    h.Position(3:4) = [600 400];
    centerfig(h);
    hold all
    imagesc(x_v, y_v, Z);
    colorbar;
    plot(x(h_idx),y(h_idx),'k--');
    axis([min(x), max(x), min(y), max(y)]);
end


end
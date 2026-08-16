function plot_radar_chart(scores, model_names, metric_names, target_name, out_dir)
% PLOT_RADAR_CHART  Radar (spider) chart comparing models across metrics.
%   scores      : n_models x n_metrics matrix, each value in [0,1] (1 = best)
%   model_names : 1 x n_models cell array
%   metric_names: 1 x n_metrics cell array

    n_models = size(scores, 1);
    n_metrics = size(scores, 2);
    colors = model_palette(n_models);
    font_name = 'Times New Roman';

    theta = linspace(0, 360, n_metrics + 1);
    theta = theta(1:end-1);
    theta_rad = deg2rad(theta);

    fig = figure('Position', [100 100 680 620], 'Color', 'w');
    pax = polaraxes(fig);
    hold(pax, 'on');
    for m = 1:n_models
        polarplot(pax, [theta_rad, theta_rad(1)], [scores(m, :), scores(m, 1)], ...
            '-o', 'Color', colors(m, :), 'LineWidth', 1.6, 'MarkerSize', 4, ...
            'MarkerFaceColor', colors(m, :));
    end
    pax.ThetaTick = theta;
    pax.ThetaTickLabel = metric_names;
    pax.ThetaZeroLocation = 'top';
    pax.ThetaDir = 'clockwise';
    pax.RLim = [0 1];
    pax.RTick = [0.25 0.5 0.75 1];
    set(pax, 'FontName', font_name, 'FontSize', 12, 'LineWidth', 1.0);
    title(pax, target_name, 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
    legend(pax, model_names, 'Location', 'eastoutside', 'FontName', font_name, 'FontSize', 10);

    export_figure(fig, fullfile(out_dir, ['Fig_radar_' target_name]));
    if isvalid(fig); close(fig); end
end

function colors = model_palette(n)
    full = [0.20 0.45 0.75; 0.85 0.33 0.30; 0.30 0.65 0.40; ...
            0.75 0.45 0.15; 0.55 0.30 0.65; 0.20 0.60 0.60; ...
            0.45 0.45 0.45; 0.90 0.55 0.70];
    colors = full(1:n, :);
end

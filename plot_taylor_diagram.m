function plot_taylor_diagram(stats, model_names, target_name, out_dir)
% PLOT_TAYLOR_DIAGRAM  Taylor diagram of model performance.
%   stats : n_models x 3 matrix [correlation R, std ratio, normalised RMSE]
%
%   Radial distance = normalised standard deviation.
%   Angle = arccos(correlation R).
%   Concentric arcs = normalised RMSE.

    n_models = size(stats, 1);
    colors = model_palette(n_models);
    font_name = 'Times New Roman';

    maxr = max(1.6, 1.3 * max([stats(:, 2); stats(:, 3)]));

    fig = figure('Position', [100 100 780 680], 'Color', 'w');
    ax = axes(fig);
    hold(ax, 'on');
    axis(ax, 'equal');

    theta = linspace(0, pi/2, 200);

    % Correlation radial lines.
    for Rc = [0 0.2 0.4 0.6 0.8 0.9 0.95 1]
        ang = acos(Rc);
        plot(ax, [0 maxr*cos(ang)], [0 maxr*sin(ang)], ':', ...
            'Color', [0.75 0.75 0.75], 'LineWidth', 0.8);
        text(ax, maxr*cos(ang)*1.03, maxr*sin(ang)*1.03, num2str(Rc), ...
            'FontSize', 9, 'FontName', font_name, 'Color', [0.35 0.35 0.35]);
    end

    % Standard-deviation concentric arcs.
    for sr = 0.25:0.25:maxr
        plot(ax, sr*cos(theta), sr*sin(theta), ':', ...
            'Color', [0.75 0.75 0.75], 'LineWidth', 0.8);
        if abs(sr - round(sr*2)/2) < 1e-6
            text(ax, sr*cos(pi/4), sr*sin(pi/4), num2str(sr), ...
                'FontSize', 9, 'FontName', font_name, 'Color', [0.35 0.35 0.35]);
        end
    end

    % RMSE contour arcs.
    for c = 0.25:0.25:maxr
        inner = c^2 - sin(theta).^2;
        valid = inner >= 0;
        s = cos(theta) + sqrt(max(inner, 0));
        s(~valid) = NaN;
        plot(ax, s.*cos(theta), s.*sin(theta), '-', ...
            'Color', [0.60 0.60 0.60], 'LineWidth', 0.6);
        idx = find(valid, 1, 'last');
        if ~isempty(idx) && ~isnan(s(idx))
            text(ax, s(idx)*cos(theta(idx)) + 0.02, s(idx)*sin(theta(idx)) + 0.02, ...
                num2str(c), 'FontSize', 8, 'FontName', font_name, 'Color', [0.45 0.45 0.45]);
        end
    end

    % Reference point (truth).
    plot(ax, 1, 0, 'k*', 'MarkerSize', 14);
    text(ax, 1, 0.08, 'REF', 'HorizontalAlignment', 'center', ...
        'FontSize', 10, 'FontName', font_name, 'FontWeight', 'bold');

    % Model points.
    for m = 1:n_models
        R = max(min(stats(m, 1), 1), -1);
        ang = acos(R);
        x = stats(m, 2) * cos(ang);
        y = stats(m, 2) * sin(ang);
        plot(ax, x, y, 'o', 'Color', colors(m, :), 'MarkerFaceColor', colors(m, :), ...
            'MarkerSize', 9, 'LineWidth', 1.2);
        text(ax, x + 0.03, y + 0.06, model_names{m}, ...
            'FontSize', 10, 'FontName', font_name, 'FontWeight', 'bold');
    end

    xlim([0, maxr*1.18]);
    ylim([0, maxr*1.18]);
    xlabel(ax, 'Normalised standard deviation', 'FontName', font_name, 'FontSize', 12);
    ylabel(ax, 'Normalised standard deviation', 'FontName', font_name, 'FontSize', 12);
    title(ax, target_name, 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
    set(ax, 'FontName', font_name, 'FontSize', 11, 'LineWidth', 1.0);
    ax.Box = 'off';

    export_figure(fig, fullfile(out_dir, ['Fig_taylor_' target_name]));
    if isvalid(fig); close(fig); end
end

function colors = model_palette(n)
    full = [0.20 0.45 0.75; 0.85 0.33 0.30; 0.30 0.65 0.40; ...
            0.75 0.45 0.15; 0.55 0.30 0.65; 0.20 0.60 0.60; ...
            0.45 0.45 0.45; 0.90 0.55 0.70];
    colors = full(1:n, :);
end

function plot_compare_bar(r2_matrix, model_names, target_names, out_dir)
% PLOT_COMPARE_BAR  Grouped bar chart of R2 across models and targets.
%   r2_matrix    : n_models x n_targets matrix of R2 values
%   model_names  : 1 x n_models cell array
%   target_names : 1 x n_targets cell array

    n_models = size(r2_matrix, 1);
    colors = model_palette(n_models);
    font_name = 'Times New Roman';

    fig = figure('Position', [100 100 900 560], 'Color', 'w');
    ax = axes(fig);
    b = bar(ax, r2_matrix', 'grouped');
    for m = 1:n_models
        b(m).FaceColor = colors(m, :);
        b(m).EdgeColor = 'none';
    end
    set(ax, 'XTickLabel', target_names);
    xlabel(ax, 'Target', 'FontName', font_name, 'FontSize', 13);
    ylabel(ax, 'R^2', 'FontName', font_name, 'FontSize', 13);
    title(ax, 'Model comparison (test-set R^2)', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
    legend(ax, model_names, 'Location', 'best', 'FontName', font_name, 'FontSize', 10);
    grid(ax, 'on'); ax.Box = 'on';
    set(ax, 'FontName', font_name, 'FontSize', 11, 'LineWidth', 1.0);

    export_figure(fig, fullfile(out_dir, 'Fig_R2_comparison_bar'));
    if isvalid(fig); close(fig); end
end

function colors = model_palette(n)
    full = [0.20 0.45 0.75; 0.85 0.33 0.30; 0.30 0.65 0.40; ...
            0.75 0.45 0.15; 0.55 0.30 0.65; 0.20 0.60 0.60; ...
            0.45 0.45 0.45; 0.90 0.55 0.70];
    colors = full(1:n, :);
end

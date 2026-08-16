function plot_shap_bar(shap_values, feature_names, target_name, out_dir)
% PLOT_SHAP_BAR  Horizontal bar chart of mean |SHAP| per feature (single
% target), sorted descending, most important on top.

    mean_abs = mean(abs(shap_values), 1);
    [sorted, idx] = sort(mean_abs, 'descend');
    feat_sorted = feature_names(idx);

    font_name = 'Times New Roman';
    fig = figure('Position', [100 100 820 620], 'Color', 'w');
    ax = gca;
    barh(ax, sorted, 'BarWidth', 0.7, 'FaceColor', [0.15 0.30 0.55]);
    set(ax, 'YDir', 'reverse');
    yticks(ax, 1:numel(sorted));
    yticklabels(ax, feat_sorted);
    xlabel(ax, 'Mean |SHAP| value', 'FontName', font_name, 'FontSize', 13);
    ylabel(ax, 'Feature', 'FontName', font_name, 'FontSize', 13);
    title(ax, [target_name '  SHAP feature importance'], ...
        'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
    grid(ax, 'on'); ax.Box = 'on';
    set(ax, 'FontName', font_name, 'FontSize', 11, 'LineWidth', 1.0);
    export_figure(fig, fullfile(out_dir, ['Fig_SHAP_bar_' target_name]));
    if isvalid(fig); close(fig); end
end

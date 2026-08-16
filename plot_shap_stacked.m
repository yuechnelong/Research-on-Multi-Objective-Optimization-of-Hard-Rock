function plot_shap_stacked(shap_mean_abs, target_names, feature_names, out_dir)
% PLOT_SHAP_STACKED  Stacked horizontal bar charts of the mean |SHAP|
% contribution per feature, one colour per target, plus a percentage version.
%
%   plot_shap_stacked(shap_mean_abs, target_names, feature_names, out_dir)
%
%   shap_mean_abs : M x K matrix (M features, K targets) of mean |SHAP|.

    [num_feat, num_target] = size(shap_mean_abs);
    total_shap = sum(shap_mean_abs, 2);
    [~, sort_id] = sort(total_shap, 'descend');
    shap_plot = shap_mean_abs(sort_id, :);
    feat_plot = feature_names(sort_id);
    shap_pct = shap_plot ./ (sum(shap_plot, 1) + eps) * 100;

    font_name = 'Times New Roman';

    % ---- Fig 1: mean |SHAP| stacked ----
    fig1 = figure('Position', [100 100 1050 680], 'Color', 'w');
    ax1 = axes(fig1);
    barh(ax1, shap_plot, 'stacked', 'BarWidth', 0.72);
    set(ax1, 'YDir', 'reverse');
    yticks(ax1, 1:num_feat);
    yticklabels(ax1, feat_plot);
    xlabel(ax1, 'Mean |SHAP| value', 'FontName', font_name, 'FontSize', 13);
    ylabel(ax1, 'Feature', 'FontName', font_name, 'FontSize', 13);
    title(ax1, 'Three-target SHAP feature contribution (stacked)', ...
        'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
    legend(ax1, target_names, 'Location', 'southeast', 'FontName', font_name, 'FontSize', 11);
    grid(ax1, 'on'); ax1.Box = 'on';
    set(ax1, 'FontName', font_name, 'FontSize', 11, 'LineWidth', 1.0);
    export_figure(fig1, fullfile(out_dir, 'Fig1_SHAP_stacked'));
    if isvalid(fig1); close(fig1); end

    % ---- Fig 2: percentage stacked ----
    fig2 = figure('Position', [100 100 1050 680], 'Color', 'w');
    ax2 = axes(fig2);
    barh(ax2, shap_pct, 'stacked', 'BarWidth', 0.72);
    set(ax2, 'YDir', 'reverse');
    yticks(ax2, 1:num_feat);
    yticklabels(ax2, feat_plot);
    xlabel(ax2, 'SHAP contribution ratio / %', 'FontName', font_name, 'FontSize', 13);
    ylabel(ax2, 'Feature', 'FontName', font_name, 'FontSize', 13);
    title(ax2, 'Three-target SHAP contribution ratio (stacked)', ...
        'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
    legend(ax2, target_names, 'Location', 'southeast', 'FontName', font_name, 'FontSize', 11);
    grid(ax2, 'on'); ax2.Box = 'on';
    set(ax2, 'FontName', font_name, 'FontSize', 11, 'LineWidth', 1.0);
    export_figure(fig2, fullfile(out_dir, 'Fig2_SHAP_ratio_stacked'));
    if isvalid(fig2); close(fig2); end
end

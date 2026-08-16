function plot_shap_beeswarm(shap_values, X_shap, feature_names, target_name, out_dir)
% PLOT_SHAP_BEESWARM  Bee-swarm (SHAP summary) plot: one row per feature,
% points jittered along the row and coloured by feature value (blue = low,
% red = high). Features are sorted by mean |SHAP| (most important on top),
% and a light horizontal bar overlay marks each feature's contribution
% percentage. No title is drawn.

    num_feat = size(shap_values, 2);
    cmap = shap_redblue_colormap(256);
    font_name = 'Times New Roman';

    % Sort features by mean |SHAP| descending.
    mean_abs = mean(abs(shap_values), 1);
    [sortedImp, idx] = sort(mean_abs, 'descend');
    feat_sorted = feature_names(idx);
    shap_sorted = shap_values(:, idx);
    X_sorted = X_shap(:, idx);

    pct = sortedImp / (sum(sortedImp) + eps) * 100;

    fig = figure('Position', [100 100 900 680], 'Color', 'w');

    % ---- bee-swarm axes ----
    ax = axes('Position', [0.16 0.12 0.55 0.76]);
    hold(ax, 'on');
    for f = 1:num_feat
        shap_col = shap_sorted(:, f);
        feat_val = X_sorted(:, f);
        row = num_feat - f + 1;              % most important feature on top
        jitter = 0.08 * randn(size(shap_col));
        jitter = max(min(jitter, 0.25), -0.25);
        ypos = row + jitter;
        cind = map_feature_color(feat_val, size(cmap, 1));
        scatter(ax, shap_col, ypos, 16, cmap(cind, :), 'filled', ...
            'MarkerFaceAlpha', 0.7, 'MarkerEdgeAlpha', 0);
    end
    yticks(ax, 1:num_feat);
    yticklabels(ax, fliplr(feat_sorted));
    ylim(ax, [0.3, num_feat + 0.7]);
    xlabel(ax, 'SHAP value (impact on model output)', 'FontName', font_name, 'FontSize', 13);
    ylabel(ax, 'Feature', 'FontName', font_name, 'FontSize', 13);
    grid(ax, 'on'); ax.Box = 'on';
    set(ax, 'FontName', font_name, 'FontSize', 11, 'LineWidth', 1.0);

    % ---- colourbar (shrinks ax to make room) ----
    colormap(ax, cmap);
    cb = colorbar(ax, 'eastoutside');
    cb.Label.String = 'Feature value';
    cb.Label.FontName = font_name;
    cb.Label.FontSize = 11;
    cb.Box = 'off';

    % ---- top bar overlay (mean |SHAP| + contribution percentage) ----
    pos = ax.Position;
    ax_bar = axes('Position', pos, 'Color', 'none', ...
        'XAxisLocation', 'top', 'YAxisLocation', 'right', 'YTick', [], 'Box', 'off');
    hold(ax_bar, 'on');
    bh = barh(ax_bar, 1:num_feat, sortedImp, 0.6);
    bh.FaceColor = [0.15 0.30 0.55];
    bh.FaceAlpha = 0.25;
    bh.EdgeColor = 'none';
    ylim(ax_bar, [0.3, num_feat + 0.7]);
    ax_bar.YDir = 'reverse';
    xmax = max(sortedImp);
    xlim(ax_bar, [0, xmax * 1.3]);
    for i = num_feat:-1:1
        text(ax_bar, sortedImp(i) + 0.04 * xmax, i, sprintf('%.1f%%', pct(i)), ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', ...
            'FontSize', 10, 'FontName', font_name);
    end
    xlabel(ax_bar, [target_name '  Mean |SHAP| value'], 'FontName', font_name, 'FontSize', 12);
    set(ax_bar, 'FontName', font_name, 'FontSize', 11);

    export_figure(fig, fullfile(out_dir, ['Fig_beeswarm_' target_name]));
    if isvalid(fig); close(fig); end
end

function cind = map_feature_color(feat_val, n)
% Map feature values to 1..n colour indices (min -> 1, max -> n).
    xmin = min(feat_val);
    xmax = max(feat_val);
    if xmax == xmin, xmax = xmin + 1e-6; end
    cind = 1 + round((feat_val - xmin) ./ (xmax - xmin) * (n - 1));
    cind = max(1, min(n, cind));
end

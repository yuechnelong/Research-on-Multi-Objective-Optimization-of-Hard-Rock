function plot_outlier_results(X, outlier_idx, info, var_names, fig_dir)
% PLOT_OUTLIER_RESULTS  Generate the two outlier-detection figures.
%
%   Fig. 5(a) - Per-target scatter of the most-correlated feature vs. target,
%               with the fitted quadratic curve and the detected outliers.
%   Fig. 5(b) - Distribution of detected outliers (2-D PCA projection).
%
%   Each figure is exported in PNG (600 DPI), EMF (vector) and FIG formats.

    % Shared typography for a clean, publication-quality look.
    font_name = 'Times New Roman';
    fs_title  = 15;
    fs_label  = 13;
    fs_tick   = 11;
    fs_legend = 11;

    n_outliers = sum(outlier_idx);
    pct        = 100 * n_outliers / numel(outlier_idx);
    n_t        = numel(info);

    %% ---- Fig. 5(a): feature-target scatter per target ----
    fig_a = figure('Position', [100 100 1500 430], 'Color', 'w');

    for t = 1:n_t
        subplot(1, n_t, t);

        xf  = info(t).xf;
        y   = info(t).y;
        out = info(t).outlier;

        scatter(xf(~out), y(~out), 8, [0.55 0.55 0.55], 'filled', 'MarkerFaceAlpha', 0.45);
        hold on;
        scatter(xf(out), y(out), 18, 'r', 'filled', 'MarkerEdgeColor', 'k');
        plot(info(t).xgrid, info(t).ygrid, 'b--', 'LineWidth', 1.8);

        xlabel(var_names{info(t).feature}, 'FontName', font_name, 'FontSize', fs_label);
        ylabel(var_names{info(t).target},  'FontName', font_name, 'FontSize', fs_label);
        title(var_names{info(t).target},   'FontName', font_name, 'FontSize', fs_title, 'FontWeight', 'bold');
        grid on; box on;
        set(gca, 'FontName', font_name, 'FontSize', fs_tick, 'LineWidth', 1.0);

        if t == 1
            legend({'Normal', 'Outlier', 'Fit'}, ...
                'FontName', font_name, 'FontSize', fs_legend, 'Location', 'northwest');
        end
    end
    sgtitle('(a) Outlier detection by binning and Isolation Forest', ...
        'FontName', font_name, 'FontSize', fs_title + 1, 'FontWeight', 'bold');
    export_figure(fig_a, fullfile(fig_dir, 'Fig5a_outlier_detection'));
    close(fig_a);

    %% ---- Fig. 5(b): Distribution of detected outliers (PCA projection) ----
    fig_b = figure('Position', [100 100 860 680], 'Color', 'w');

    Xz = (X - mean(X, 1)) ./ std(X, 0, 1);
    [~, score_pc] = pca(Xz, 'NumComponents', 2);

    scatter(score_pc(~outlier_idx, 1), score_pc(~outlier_idx, 2), 14, ...
        [0.55 0.55 0.55], 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    scatter(score_pc(outlier_idx, 1), score_pc(outlier_idx, 2), 26, ...
        'r', 'filled', 'MarkerEdgeColor', 'k');

    xlabel('First principal component', 'FontName', font_name, 'FontSize', fs_label);
    ylabel('Second principal component', 'FontName', font_name, 'FontSize', fs_label);
    legend({'Normal samples', sprintf('Outliers (%.1f%%)', pct)}, ...
        'FontName', font_name, 'FontSize', fs_legend, 'Location', 'northeastoutside');
    title('(b) Distribution of detected outliers', ...
        'FontName', font_name, 'FontSize', fs_title, 'FontWeight', 'bold');
    grid on; box on;
    set(gca, 'FontName', font_name, 'FontSize', fs_tick, 'LineWidth', 1.1);
    export_figure(fig_b, fullfile(fig_dir, 'Fig5b_outlier_distribution'));
    close(fig_b);

    fprintf('Figures saved to %s (PNG + EMF + FIG)\n', fig_dir);
end

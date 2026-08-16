function plot_prediction_results(test_true, test_pred, target_names, fig_dir)
% PLOT_PREDICTION_RESULTS  Scatter plot of predicted vs. true values on the
% test set, one panel per target, with the R2 value. No title is drawn.

    font_name = 'Times New Roman';
    fs_label  = 16;   % axis label font size
    fs_tick   = 14;   % tick label font size
    fs_r2     = 20;   % R2 annotation font size

    n_t = size(test_true, 2);

    fig = figure('Position', [100 100 1800 560], 'Color', 'w');
    for t = 1:n_t
        subplot(1, n_t, t);

        y_true = test_true(:, t);
        y_pred = test_pred(:, t);

        scatter(y_true, y_pred, 14, [0.20 0.45 0.75], 'filled', 'MarkerFaceAlpha', 0.55);
        hold on;

        % y = x reference line.
        lims = [min([y_true; y_pred]), max([y_true; y_pred])];
        plot(lims, lims, 'k--', 'LineWidth', 1.6);

        % R2 annotation.
        r2 = 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2);
        text(lims(1) + 0.06*(lims(2) - lims(1)), lims(2) - 0.10*(lims(2) - lims(1)), ...
            sprintf('R^2 = %.4f', r2), 'FontName', font_name, 'FontSize', fs_r2, ...
            'FontWeight', 'bold');

        xlabel(['True ', target_names{t}], 'FontName', font_name, 'FontSize', fs_label);
        ylabel(['Predicted ', target_names{t}], 'FontName', font_name, 'FontSize', fs_label);
        grid on; box on; axis square;
        set(gca, 'FontName', font_name, 'FontSize', fs_tick, 'LineWidth', 1.1);
    end

    export_figure(fig, fullfile(fig_dir, 'Fig_prediction_scatter'));
    if isvalid(fig); close(fig); end
end

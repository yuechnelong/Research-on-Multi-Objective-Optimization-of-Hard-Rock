%% main_3_model_predict.m
% Step 3 - RF model prediction with Bayesian hyperparameter optimisation.

clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(script_dir, '2.3_model_prediction');

short_names = {'GP1','GP5','GP6','CS','TPA','TPB','TPC','TPD','TPE','TPF','AR','SCS','PPR','CWP','TCP'};
feature_idx = 1:12;
target_idx  = 13:15;
top_k       = 8;
num_BO_iter = 30;

feature_names_all = short_names(feature_idx);
target_names      = short_names(target_idx);

if ~exist(out_dir, 'dir'); mkdir(out_dir); end

%% 1. Load split data + selected features (from step 2)
data = load(fullfile(script_dir, 'data_split.mat'));
X_train = data.X_train; X_valid = data.X_valid; X_test = data.X_test;

S = load(fullfile(script_dir, 'selected_features.mat'));
sel_idx = S.sel_idx;
feature_names = S.selected_names;
fprintf('Selected %d features:', numel(sel_idx));
fprintf(' %s', feature_names{:});
fprintf('\n');

Xtr_f = X_train(:, sel_idx);
Xva_f = X_valid(:, sel_idx);
Xte_f = X_test (:, sel_idx);

%% 2. Per-target Bayesian optimisation + training + evaluation
set_names = {'Train', 'Valid', 'Test'};
metric_names = {'Target','Set','MAE','MAPE','MSE','RMSE','R2','NRMSE','NMAE'};
metric_rows = cell(0, numel(metric_names));
best_params_all = zeros(3, 3);
conv_all = zeros(num_BO_iter, 3);

test_true = zeros(size(X_test,1), 3);
test_pred = zeros(size(X_test,1), 3);

for t = 1:3
    ytr = X_train(:, target_idx(t));
    yva = X_valid(:, target_idx(t));
    yte = X_test (:, target_idx(t));

    [mdl, conv] = optimizebaye_fitrTreeBagger(Xtr_f, ytr, Xva_f, yva, num_BO_iter);
    conv_all(:, t) = conv(:);
    best_params_all(t, 1) = mdl.NumTrees;
    best_params_all(t, 2) = mdl.MinLeafSize;
    best_params_all(t, 3) = mdl.NumPredictorsToSample;

    p_tr = predict(mdl, Xtr_f);
    p_va = predict(mdl, Xva_f);
    p_te = predict(mdl, Xte_f);
    test_true(:, t) = yte;
    test_pred(:, t) = p_te;

    sets_true = {ytr, yva, yte};
    sets_pred = {p_tr, p_va, p_te};
    for s = 1:3
        y = sets_true{s}; p = sets_pred{s};
        mae  = mean(abs(p - y));
        mape = mean(abs((p - y) ./ (abs(y) + eps)));
        mse  = mean((p - y).^2);
        rmse = sqrt(mse);
        r2   = 1 - sum((y - p).^2) / sum((y - mean(y)).^2);
        nrmse = rmse / std(y);
        nmae  = mae  / std(y);
        metric_rows(end+1, :) = {target_names{t}, set_names{s}, mae, mape, mse, rmse, r2, nrmse, nmae};
    end
end

%% 3. Overall balanced metrics
std_true = std(test_true);
r2_per = 1 - sum((test_true - test_pred).^2) ./ sum((test_true - mean(test_true)).^2);
overall_r2    = mean(r2_per);
overall_nrmse = mean(sqrt(mean((test_pred - test_true).^2)) ./ std_true);
overall_nmae  = mean(mean(abs(test_pred - test_true)) ./ std_true);
overall_mae   = mean(mean(abs(test_pred - test_true)));
overall_mape  = mean(mean(abs((test_pred - test_true) ./ (abs(test_true) + eps))));
overall_rmse  = mean(sqrt(mean((test_pred - test_true).^2)));
fprintf('\nBalanced overall: R2=%.4f, NRMSE=%.4f, NMAE=%.4f\n', overall_r2, overall_nrmse, overall_nmae);

%% 4. Figures
plot_prediction_results(test_true, test_pred, target_names, out_dir);

conv_color = [160 123 194] ./ 255;
for t = 1:3
    fig_conv = figure('Position', [100 100 780 600], 'Color', 'w');
    plot(conv_all(:, t), '--p', 'LineWidth', 1.2, 'Color', conv_color);
    xticks(1:num_BO_iter);
    xlabel('iter'); ylabel('fitness');
    grid on; box off;
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 14, 'LineWidth', 1.1);
    export_figure(fig_conv, fullfile(out_dir, ['Fig_bayesian_convergence_' target_names{t}]));
    if isvalid(fig_conv); close(fig_conv); end
end

%% 5. Export tables (delete first)
files = {fullfile(out_dir, 'per_target_metrics.xlsx'), fullfile(out_dir, 'bayesian_hyperparameters.xlsx'), fullfile(out_dir, 'overall_metrics.xlsx')};
for i = 1:numel(files), if exist(files{i}, 'file'), delete(files{i}); end; end

metric_table = cell2table(metric_rows, 'VariableNames', metric_names);
writetable(metric_table, files{1});

param_table = array2table(best_params_all, 'VariableNames', {'NumTrees','MinLeafSize','NumPredictorsToSample'}, 'RowNames', target_names);
writetable(param_table, files{2}, 'WriteRowNames', true);

overall_table = table(overall_r2, overall_nrmse, overall_nmae, overall_mae, overall_mape, overall_rmse, ...
    'VariableNames', {'OverallR2','OverallNRMSE','OverallNMAE','OverallMAE','OverallMAPE','OverallRMSE'});
writetable(overall_table, files{3});

% Predicted vs true values (test set)
f4 = fullfile(out_dir, 'predictions_test.xlsx');
if exist(f4, 'file'), delete(f4); end
pred_true = array2table([test_true, test_pred], ...
    'VariableNames', [strcat('True_', target_names), strcat('Pred_', target_names)]);
writetable(pred_true, f4);

fprintf('Step 3 completed.\n');

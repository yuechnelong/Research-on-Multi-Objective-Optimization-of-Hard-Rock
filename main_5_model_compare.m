%% main_5_model_compare.m
% Step 5 - Multi-algorithm model comparison (RF vs SVR/GBDT/KNN/DT/LR).

clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(script_dir, '2.5_model_comparison');
out_23  = fullfile(script_dir, '2.3_model_prediction');

short_names = {'GP1','GP5','GP6','CS','TPA','TPB','TPC','TPD','TPE','TPF','AR','SCS','PPR','CWP','TCP'};
feature_idx = 1:12;
target_idx  = 13:15;
model_names = {'RF','SVR','GBDT','KNN','DT','LR'};
metric_names = {'R^2','MAE','RMSE','MAPE'};
target_names = short_names(target_idx);

if ~exist(out_dir, 'dir'); mkdir(out_dir); end

%% 1. Load data + selected features
data = load(fullfile(script_dir, 'data_split.mat'));
X_train = data.X_train; X_test = data.X_test;

S = load(fullfile(script_dir, 'selected_features.mat'));
sel_idx = S.sel_idx;

Xtr_f = X_train(:, sel_idx);
Xte_f = X_test (:, sel_idx);

%% 2. RF hyperparameters (from step 3)
params = readtable(fullfile(out_23, 'bayesian_hyperparameters.xlsx'), 'ReadRowNames', true);
numTrees_all = params.NumTrees; minLeaf_all = params.MinLeafSize; numPred_all = params.NumPredictorsToSample;

%% 3. Train models + compute metrics
n_target = numel(target_idx);
n_models = numel(model_names);
metrics = zeros(n_models, n_target, 6);        % R2, MAE, RMSE, MAPE, NRMSE, NMAE
taylor_stats = zeros(n_models, n_target, 3);   % corr, std ratio, nrmse

for t = 1:n_target
    ytr = X_train(:, target_idx(t));
    yte = X_test (:, target_idx(t));

    % RF (proposed)
    mdl = TreeBagger(numTrees_all(t), Xtr_f, ytr, 'Method', 'regression', ...
        'MinLeafSize', minLeaf_all(t), 'NumPredictorsToSample', numPred_all(t));
    preds{1} = predict(mdl, Xte_f);

    % SVR
    mdl = fitrsvm(Xtr_f, ytr, 'KernelFunction', 'gaussian', 'Standardize', true);
    preds{2} = predict(mdl, Xte_f);

    % GBDT
    mdl = fitrensemble(Xtr_f, ytr, 'Method', 'LSBoost', 'NumLearningCycles', 100, ...
        'Learners', templateTree('MaxNumSplits', 10));
    preds{3} = predict(mdl, Xte_f);

    % KNN (manual knnsearch)
    mu_f = mean(Xtr_f); sig_f = std(Xtr_f); sig_f(sig_f == 0) = 1;
    knnIdx = knnsearch((Xtr_f - mu_f) ./ sig_f, (Xte_f - mu_f) ./ sig_f, 'K', 5);
    preds{4} = mean(ytr(knnIdx), 2);

    % DT
    mdl = fitrtree(Xtr_f, ytr);
    preds{5} = predict(mdl, Xte_f);

    % LR
    mdl = fitlm(Xtr_f, ytr);
    preds{6} = predict(mdl, Xte_f);

    for m = 1:n_models
        p = preds{m};
        sy = std(yte);
        metrics(m, t, 1) = 1 - sum((yte - p).^2) / sum((yte - mean(yte)).^2);
        metrics(m, t, 2) = mean(abs(p - yte));
        metrics(m, t, 3) = sqrt(mean((p - yte).^2));
        metrics(m, t, 4) = mean(abs((p - yte) ./ (abs(yte) + eps)));
        metrics(m, t, 5) = metrics(m, t, 3) / sy;
        metrics(m, t, 6) = metrics(m, t, 2) / sy;
        R = corr(p, yte); if isnan(R), R = 0; end
        taylor_stats(m, t, 1) = R;
        taylor_stats(m, t, 2) = std(p) / sy;
        taylor_stats(m, t, 3) = metrics(m, t, 5);
    end
end

%% 4. Figures
for t = 1:n_target
    R2 = metrics(:, t, 1); MAE = metrics(:, t, 2); RMSE = metrics(:, t, 3); MAPE = metrics(:, t, 4);
    scores = [norm_best(R2), norm_best(-MAE), norm_best(-RMSE), norm_best(-MAPE)];
    plot_radar_chart(scores, model_names, metric_names, target_names{t}, out_dir);
    plot_taylor_diagram(squeeze(taylor_stats(:, t, :)), model_names, target_names{t}, out_dir);
end
plot_compare_bar(metrics(:, :, 1), model_names, target_names, out_dir);

%% 5. Export tables (delete first)
files = {fullfile(out_dir,'model_comparison_metrics.xlsx'), fullfile(out_dir,'overall_comparison.xlsx'), fullfile(out_dir,'R2_matrix.xlsx')};
for i = 1:numel(files), if exist(files{i},'file'), delete(files{i}); end; end

metric_rows = cell(0, 8);
for t = 1:n_target
    for m = 1:n_models
        metric_rows(end+1, :) = {target_names{t}, model_names{m}, metrics(m,t,1), metrics(m,t,2), metrics(m,t,4), metrics(m,t,3), metrics(m,t,5), metrics(m,t,6)};
    end
end
metric_table = cell2table(metric_rows, 'VariableNames', {'Target','Model','R2','MAE','MAPE','RMSE','NRMSE','NMAE'});
writetable(metric_table, files{1});

overall_rows = cell(0, 4);
for m = 1:n_models
    overall_rows(end+1, :) = {model_names{m}, mean(metrics(m,:,1)), mean(metrics(m,:,5)), mean(metrics(m,:,6))};
end
overall_table = cell2table(overall_rows, 'VariableNames', {'Model','OverallR2','OverallNRMSE','OverallNMAE'});
writetable(overall_table, files{2});

r2_table = array2table(metrics(:, :, 1), 'VariableNames', target_names, 'RowNames', model_names);
writetable(r2_table, files{3}, 'WriteRowNames', true);

fprintf('Step 5 completed.\n');

function x = norm_best(v)
    vmin = min(v); vmax = max(v);
    if vmax == vmin, x = zeros(size(v)); else, x = (v - vmin) / (vmax - vmin); end
end

%% main_4_shap_analysis.m
% Step 4 - SHAP interpretation of the trained Random Forests.

clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(script_dir, '2.4_shap_analysis');
out_23  = fullfile(script_dir, '2.3_model_prediction');

short_names = {'GP1','GP5','GP6','CS','TPA','TPB','TPC','TPD','TPE','TPF','AR','SCS','PPR','CWP','TCP'};
feature_idx = 1:12;
target_idx  = 13:15;
top_k = 8;
num_shap_sample = 200;
rng_seed = 0;

feature_names_all = short_names(feature_idx);
target_names      = short_names(target_idx);

if ~exist(out_dir, 'dir'); mkdir(out_dir); end

%% 1. Load data + selected features + train models
data = load(fullfile(script_dir, 'data_split.mat'));
X_train = data.X_train; X_test = data.X_test;

S = load(fullfile(script_dir, 'selected_features.mat'));
sel_idx = S.sel_idx;
feature_names = S.selected_names;

Xtr_f = X_train(:, sel_idx);
Xte_f = X_test (:, sel_idx);

params = readtable(fullfile(out_23, 'bayesian_hyperparameters.xlsx'), 'ReadRowNames', true);
numTrees_all = params.NumTrees; minLeaf_all = params.MinLeafSize; numPred_all = params.NumPredictorsToSample;

models = cell(1, 3);
for t = 1:3
    models{t} = TreeBagger(numTrees_all(t), Xtr_f, X_train(:, target_idx(t)), ...
        'Method', 'regression', 'MinLeafSize', minLeaf_all(t), 'NumPredictorsToSample', numPred_all(t));
end

%% 2. Compute SHAP
[shap_values, X_shap, sample_idx] = compute_shap_baseline(models, Xte_f, Xtr_f, num_shap_sample, rng_seed);

shap_mean_abs = zeros(numel(feature_names), 3);
for k = 1:3
    shap_mean_abs(:, k) = mean(abs(shap_values(:, :, k)), 1)';
end

%% 3. Figures
plot_shap_stacked(shap_mean_abs, target_names, feature_names, out_dir);
for k = 1:3
    plot_shap_beeswarm(shap_values(:, :, k), X_shap, feature_names, target_names{k}, out_dir);
    plot_shap_bar(shap_values(:, :, k), feature_names, target_names{k}, out_dir);
end

%% 4. Export tables
files = {fullfile(out_dir, 'mean_abs_SHAP.xlsx'), fullfile(out_dir, 'per_target_SHAP.xlsx'), fullfile(out_dir, 'SHAP_sample_features.xlsx')};
for i = 1:numel(files), if exist(files{i}, 'file'), delete(files{i}); end; end

total_contribution = sum(shap_mean_abs, 2);
shap_table = array2table([shap_mean_abs, total_contribution], ...
    'VariableNames', [target_names, {'TotalContribution'}], 'RowNames', feature_names);
writetable(shap_table, files{1}, 'WriteRowNames', true);

for k = 1:3
    sheet_name = ['SHAP_', target_names{k}];
    shap_k = shap_values(:, :, k);
    tbl = array2table([sample_idx(:), shap_k], 'VariableNames', [{'SampleIndex'}, feature_names]);
    writetable(tbl, files{2}, 'Sheet', sheet_name);
end

sample_tbl = array2table([sample_idx(:), X_shap], 'VariableNames', [{'SampleIndex'}, feature_names]);
writetable(sample_tbl, files{3});

fprintf('Step 4 completed.\n');

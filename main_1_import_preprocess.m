%% main_1_import_preprocess.m
% Step 1 - Data import, split and Isolation-Forest outlier removal.
% Outlier detection and feature selection are done on Train + Valid ONLY;
% the test set is held out entirely (no data leakage).

clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(script_dir, '1.0_data_import');

short_names = {'GP1','GP5','GP6','CS','TPA','TPB','TPC','TPD','TPE','TPF','AR','SCS','PPR','CWP','TCP'};
feature_idx = 1:12;
target_idx  = 13:15;

if ~exist(out_dir, 'dir'); mkdir(out_dir); end

%% 1. Read data.xlsx (Set | 12 features | 3 targets)
T = readtable(fullfile(script_dir, 'data.xlsx'), 'VariableNamingRule', 'preserve');
set_col  = T.Set;
data_all = [table2array(T(:, 2:13)), table2array(T(:, 14:16))];

%% 2. Split: hold out the test set entirely
train_valid_mask = strcmp(set_col, 'Train') | strcmp(set_col, 'Valid');
test_mask  = strcmp(set_col, 'Test');

train_valid     = data_all(train_valid_mask, :);
train_valid_set = set_col(train_valid_mask);
X_test = data_all(test_mask, :);

%% 3. Isolation Forest on Train + Valid only (no test leakage)
x = zscore(train_valid);
[Mdl, tf, scores] = iforest(x, 'ContaminationFraction', 0.05);
fprintf('Outliers removed: %d / %d (%.2f%%)\n', sum(tf), size(train_valid,1), 100*sum(tf)/size(train_valid,1));

train_valid_clean     = train_valid(~tf, :);
train_valid_set_clean = train_valid_set(~tf);

X_train = train_valid_clean(strcmp(train_valid_set_clean, 'Train'), :);
X_valid = train_valid_clean(strcmp(train_valid_set_clean, 'Valid'), :);
fprintf('Train=%d, Valid=%d, Test=%d\n', size(X_train,1), size(X_valid,1), size(X_test,1));

%% 4. Figures + save
plot_iforest_results(x, tf, scores, Mdl, out_dir);

save(fullfile(script_dir, 'data_split.mat'), 'X_train', 'X_valid', 'X_test');
fprintf('Step 1 completed. Saved data_split.mat\n');

%% ==================== Local function ====================
function plot_iforest_results(x, tf, scores, Mdl, out_dir)
    font_name = 'Times New Roman';

    % Fig (a): distribution of anomaly scores.
    fig1 = figure('Position', [100 100 820 560], 'Color', 'w');
    histogram(scores, 40, 'FaceColor', [0.20 0.45 0.75], 'EdgeColor', 'none');
    hold on;
    xline(Mdl.ScoreThreshold, 'r-', 'LineWidth', 2);
    xlabel('Anomaly score', 'FontName', font_name, 'FontSize', 16);
    ylabel('Count', 'FontName', font_name, 'FontSize', 16);
    grid on; box on;
    set(gca, 'FontName', font_name, 'FontSize', 14, 'LineWidth', 1.1);
    export_figure(fig1, fullfile(out_dir, 'Fig1_anomaly_score_distribution'));
    if isvalid(fig1); close(fig1); end

    % Fig (b): distribution of detected outliers (t-SNE).
    T = tsne(x, 'Standardize', true);
    fig2 = figure('Position', [100 100 860 600], 'Color', 'w');
    gscatter(T(:,1), T(:,2), tf, 'br', [], 10, 'off');
    legend({'Normal', 'Outlier'}, 'Location', 'best', 'FontName', font_name, 'FontSize', 14);
    grid on; box on;
    xlabel('dimension1', 'FontName', font_name, 'FontSize', 16);
    ylabel('dimension2', 'FontName', font_name, 'FontSize', 16);
    set(gca, 'FontName', font_name, 'FontSize', 14, 'LineWidth', 1.1);
    export_figure(fig2, fullfile(out_dir, 'Fig2_outlier_distribution'));
    if isvalid(fig2); close(fig2); end
end

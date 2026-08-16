%% main_2_feature_select.m
% Step 2 - Feature selection via three-target comprehensive Relief.
% Computed on Train + Valid ONLY (the test set is held out, no leakage).

clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(script_dir, '2.1_feature_selection');

short_names = {'GP1','GP5','GP6','CS','TPA','TPB','TPC','TPD','TPE','TPF','AR','SCS','PPR','CWP','TCP'};
feature_idx = 1:12;
target_idx  = 13:15;
top_k       = 8;

feature_names = short_names(feature_idx);
target_names  = short_names(target_idx);

if ~exist(out_dir, 'dir'); mkdir(out_dir); end

%% 1. Load Train + Valid (test set excluded)
data = load(fullfile(script_dir, 'data_split.mat'));
train_valid = [data.X_train; data.X_valid];
X_features = train_valid(:, feature_idx);
Y_targets  = train_valid(:, target_idx);
fprintf('Loaded Train+Valid: %d samples x %d variables\n', size(X_features,1), size(X_features,2));

%% 2. Three-target comprehensive Relief
[weights_avg, weights_each] = relief_feature_select(X_features, Y_targets, 10);
[~, sort_idx] = sort(weights_avg, 'descend');
sel_idx = sort_idx(1:top_k);
selected_names = feature_names(sel_idx);
fprintf('Selected %d features:', top_k);
fprintf(' %s', selected_names{:});
fprintf('\n');

% Save the selected feature indices/names for the downstream steps.
save(fullfile(script_dir, 'selected_features.mat'), 'sel_idx', 'selected_names');

%% 3. Export tables (delete first to avoid stale rows)
f1 = fullfile(out_dir, 'relief_importance.xlsx');
if exist(f1, 'file'), delete(f1); end
imp_table = array2table([weights_each', weights_avg'], ...
    'VariableNames', [target_names, {'Averaged'}], 'RowNames', feature_names);
writetable(imp_table, f1, 'WriteRowNames', true);

f2 = fullfile(out_dir, 'selected_features.xlsx');
if exist(f2, 'file'), delete(f2); end
sel_table = table(selected_names', 'VariableNames', {'SelectedFeature'});
writetable(sel_table, f2);

%% 4. Stacked bar chart
plot_stacked_importance(weights_each, weights_avg, feature_names, target_names, out_dir);

fprintf('Step 2 completed.\n');

%% ==================== Local function ====================
function plot_stacked_importance(weights_each, weights_avg, feature_names, target_names, out_dir)
    [~, sort_idx] = sort(weights_avg, 'descend');
    W = weights_each(:, sort_idx)';
    feat_sorted = feature_names(sort_idx);

    colors = [0.20 0.45 0.75; 0.85 0.60 0.20; 0.30 0.65 0.40];
    font_name = 'Times New Roman';

    fig = figure('Position', [100 100 1000 760], 'Color', 'w');
    ax = axes(fig);
    b = barh(ax, W, 'stacked', 'BarWidth', 0.72);
    for k = 1:numel(target_names)
        b(k).FaceColor = colors(k, :);
        b(k).EdgeColor = 'none';
    end
    set(ax, 'YDir', 'reverse');
    yticks(ax, 1:numel(feat_sorted));
    yticklabels(ax, feat_sorted);
    xlabel(ax, 'Normalised Relief importance', 'FontName', font_name, 'FontSize', 16);
    ylabel(ax, 'Feature', 'FontName', font_name, 'FontSize', 16);
    legend(ax, target_names, 'Location', 'southeast', 'FontName', font_name, 'FontSize', 14);
    grid(ax, 'on'); ax.Box = 'on';
    set(ax, 'FontName', font_name, 'FontSize', 14, 'LineWidth', 1.1);
    export_figure(fig, fullfile(out_dir, 'Fig_Relief_importance_stacked'));
    if isvalid(fig); close(fig); end
end

%% main_5_multi_objective.m
% Step 5 - Multi-objective optimisation of TBM operating parameters.
% Uses NSGA-II, MOEA/D and MOPSO (same population and iterations), with the
% Relief-selected features as decision variables and a Mahalanobis-distance
% feasibility constraint.

clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(script_dir, '2.6_multiobjective');
out_23  = fullfile(script_dir, '2.3_model_prediction');

short_names = {'GP1','GP5','GP6','CS','TPA','TPB','TPC','TPD','TPE','TPF','AR','SCS','PPR','CWP','TCP'};
feature_idx = 1:12;
target_idx  = 13:15;
top_k = 8;

pop_size = 100;
max_iter = 30;
feas_pct = 99;

feature_names_all = short_names(feature_idx);
target_names      = short_names(target_idx);

if ~exist(out_dir, 'dir'); mkdir(out_dir); end

%% 1. Load data + selected features (decision variables from step 2)
data = load(fullfile(script_dir, 'data_split.mat'));
X_train = data.X_train;

S = load(fullfile(script_dir, 'selected_features.mat'));
sel_idx = S.sel_idx;
feature_names = S.selected_names;
fprintf('Decision variables (%d):', numel(sel_idx));
fprintf(' %s', feature_names{:});
fprintf('\n');

Xtr_f = X_train(:, sel_idx);

%% 2. Train RF surrogates (Bayesian hyperparameters)
params = readtable(fullfile(out_23, 'bayesian_hyperparameters.xlsx'), 'ReadRowNames', true);
numTrees_all = params.NumTrees; minLeaf_all = params.MinLeafSize; numPred_all = params.NumPredictorsToSample;

models = cell(1, 3);
for t = 1:3
    models{t} = TreeBagger(numTrees_all(t), Xtr_f, X_train(:, target_idx(t)), ...
        'Method', 'regression', 'MinLeafSize', minLeaf_all(t), 'NumPredictorsToSample', numPred_all(t));
end

%% 3. Search bounds (5th/95th percentile)
lb = prctile(Xtr_f, 5, 1);
ub = prctile(Xtr_f, 95, 1);
for i = 1:numel(lb)
    if lb(i) == ub(i), lb(i) = min(Xtr_f(:,i)); ub(i) = max(Xtr_f(:,i)); end
    if lb(i) == ub(i), lb(i) = lb(i) - 1e-6; ub(i) = ub(i) + 1e-6; end
end
num_var = numel(lb);

%% 4. Feasibility: Mahalanobis distance
feas_mu = mean(Xtr_f, 1);
feas_inv_cov = pinv(cov(Xtr_f));
d_train = sqrt(sum(((Xtr_f - feas_mu) * feas_inv_cov) .* (Xtr_f - feas_mu), 2));
feas_threshold = prctile(d_train, feas_pct);

%% 5. Optimisation
rng(0);
objfun = @(x) shield_multi_objective_penalized(x, models, feas_mu, feas_inv_cov, feas_threshold);

fprintf('\nRunning NSGA-II ...\n');
[x_nsga, ~] = run_nsga2(objfun, lb, ub, pop_size, max_iter);
fprintf('Running MOEA/D ...\n');
[x_moead, ~] = run_moead(objfun, lb, ub, pop_size, max_iter, 20);
fprintf('Running MOPSO ...\n');
[x_mopso, ~] = run_mopso(objfun, lb, ub, pop_size, max_iter, 100);

%% 6. Merge + true objectives + feasibility filter
X_all = [x_nsga; x_moead; x_mopso];
alg_all = [repmat({'NSGA-II'}, size(x_nsga,1), 1);
           repmat({'MOEA/D'}, size(x_moead,1), 1);
           repmat({'MOPSO'},  size(x_mopso,1), 1)];

Y_all = zeros(size(X_all,1), 3);
d_all = zeros(size(X_all,1), 1);
for i = 1:size(X_all,1)
    y = zeros(1,3);
    for k = 1:3, y(k) = predict(models{k}, X_all(i,:)); end
    Y_all(i,:) = [y(1), y(2), y(3)];
    dx = X_all(i,:) - feas_mu;
    d_all(i) = sqrt(dx * feas_inv_cov * dx');
end

feasible = d_all <= feas_threshold;
X_feas = X_all(feasible,:); Y_feas = Y_all(feasible,:); alg_feas = alg_all(feasible);

F_feas = [-Y_feas(:,1), Y_feas(:,2), Y_feas(:,3)];
nd_flag = get_nondominated_flag(F_feas);
X_nd = X_feas(nd_flag,:); Y_nd = Y_feas(nd_flag,:); alg_nd = alg_feas(nd_flag);
fprintf('\nMerged: %d, feasible: %d, non-dominated: %d\n', size(X_all,1), size(X_feas,1), size(X_nd,1));

%% 7. Figures
plot_pareto3d(Y_all, alg_all, out_dir, 'Fig1_three_alg_pareto');
plot_pareto3d(Y_nd, alg_nd, out_dir, 'Fig2_nondominated_pareto');
plot_pareto2d(Y_nd, out_dir);

%% 8. Export tables (delete first)
files = {fullfile(out_dir,'variable_bounds.xlsx'), fullfile(out_dir,'pareto_solutions.xlsx'), fullfile(out_dir,'nondominated_solutions.xlsx'), fullfile(out_dir,'feasibility_summary.xlsx')};
for i = 1:numel(files), if exist(files{i},'file'), delete(files{i}); end; end

bound_cell = [{'Variable','Lower_5pct','Upper_95pct'}; [feature_names(:), num2cell(lb(:)), num2cell(ub(:))]];
writecell(bound_cell, files{1});

header = [{'Algorithm'}, feature_names, {'PPR','CWP','TCP','MahalanobisD','Feasible','NonDominated'}];
feas_cell = [header; [alg_all, num2cell([X_all, Y_all, d_all]), num2cell(feasible), num2cell(nd_flag)]];
writecell(feas_cell, files{2});

d_nd = zeros(size(X_nd,1), 1);
for i = 1:size(X_nd,1)
    dx = X_nd(i,:) - feas_mu; d_nd(i) = sqrt(dx * feas_inv_cov * dx');
end
nd_cell = [header; [alg_nd, num2cell([X_nd, Y_nd, d_nd]), num2cell(true(size(X_nd,1),1)), num2cell(true(size(X_nd,1),1))]];
writecell(nd_cell, files{3});

summary = {'Feasibility metric','Value'; 'Mahalanobis threshold', feas_threshold; 'Total merged', size(X_all,1); 'Feasible', size(X_feas,1); 'Non-dominated', size(X_nd,1)};
writecell(summary, files{4});

fprintf('Step 5 completed.\n');

%% ==================== Local functions ====================
function f = shield_multi_objective_penalized(x, models, feas_mu, feas_inv_cov, feas_threshold)
    f = [0,0,0];
    for k = 1:3, f(k) = predict(models{k}, x); end
    f = [-f(1), f(2), f(3)];   % maximise PPR, minimise CWP, TCP
    dx = x - feas_mu;
    d = sqrt(dx * feas_inv_cov * dx');
    if d > feas_threshold, f = f + 1e6 * (d - feas_threshold); end
end

function plot_pareto3d(Y, alg, out_dir, figname)
    alg_names = unique(alg);
    colors = [0.20 0.45 0.75; 0.85 0.33 0.30; 0.30 0.65 0.40];
    font_name = 'Times New Roman';
    fig = figure('Position', [100 100 900 720], 'Color', 'w');
    ax = axes(fig); hold(ax, 'on');
    hs = gobjects(numel(alg_names), 1);
    for a = 1:numel(alg_names)
        idx = strcmp(alg, alg_names{a});
        hs(a) = scatter3(ax, Y(idx,1), Y(idx,2), Y(idx,3), 40, colors(a,:), 'filled', 'MarkerFaceAlpha', 0.7, 'MarkerEdgeAlpha', 0.6);
    end
    xlabel(ax, 'PPR (penetration rate)  \uparrow', 'FontName', font_name, 'FontSize', 13);
    ylabel(ax, 'CWP (cutterhead wear pressure)  \downarrow', 'FontName', font_name, 'FontSize', 13);
    zlabel(ax, 'TCP (cutterhead total power)  \downarrow', 'FontName', font_name, 'FontSize', 13);
    legend(ax, hs, alg_names, 'Location', 'best', 'FontName', font_name, 'FontSize', 11);
    grid(ax, 'on'); ax.Box = 'on'; view(ax, [135 25]);
    set(ax, 'FontName', font_name, 'FontSize', 11, 'LineWidth', 1.0);
    export_figure(fig, fullfile(out_dir, figname));
    if isvalid(fig); close(fig); end
end

function plot_pareto2d(Y, out_dir)
    font_name = 'Times New Roman';
    pairs = {[1 2], [1 3], [2 3]};
    xlabs = {'PPR \uparrow', 'PPR \uparrow', 'CWP \downarrow'};
    ylabs = {'CWP \downarrow', 'TCP \downarrow', 'TCP \downarrow'};
    fig = figure('Position', [100 100 1200 400], 'Color', 'w');
    for s = 1:3
        subplot(1, 3, s);
        scatter(Y(:, pairs{s}(1)), Y(:, pairs{s}(2)), 30, [0.20 0.45 0.75], 'filled', 'MarkerFaceAlpha', 0.7);
        xlabel(xlabs{s}, 'FontName', font_name, 'FontSize', 12);
        ylabel(ylabs{s}, 'FontName', font_name, 'FontSize', 12);
        grid on; box on;
        set(gca, 'FontName', font_name, 'FontSize', 11, 'LineWidth', 1.0);
    end
    export_figure(fig, fullfile(out_dir, 'Fig3_pareto_2d'));
    if isvalid(fig); close(fig); end
end

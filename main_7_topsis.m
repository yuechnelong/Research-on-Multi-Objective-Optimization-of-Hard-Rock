%% main_6_topsis.m
% Step 6 - TOPSIS evaluation of the non-dominated Pareto solutions.

clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(script_dir, '2.7_topsis');
out_26  = fullfile(script_dir, '2.6_multiobjective');

feature_names = {'AR','CS','TPC','TPB','TPD','TPA','TPE','TPF'};
target_names  = {'PPR','CWP','TCP'};
weight = [1/3, 1/3, 1/3];

if ~exist(out_dir, 'dir'); mkdir(out_dir); end

%% 1. Load non-dominated feasible solutions
C = readcell(fullfile(out_26, 'nondominated_solutions.xlsx'));
alg = C(2:end, 1);
X   = cell2mat(C(2:end, 2:9));     % 8 decision variables
Y   = cell2mat(C(2:end, 10:12));   % PPR, CWP, TCP
n   = size(Y, 1);
fprintf('Loaded %d non-dominated feasible solutions.\n', n);

%% 2. TOPSIS scoring
Z = zeros(n, 3);
Z(:, 1) = (Y(:, 1) - min(Y(:, 1))) ./ (max(Y(:, 1)) - min(Y(:, 1)) + eps);   % PPR benefit
Z(:, 2) = (max(Y(:, 2)) - Y(:, 2)) ./ (max(Y(:, 2)) - min(Y(:, 2)) + eps);   % CWP cost
Z(:, 3) = (max(Y(:, 3)) - Y(:, 3)) ./ (max(Y(:, 3)) - min(Y(:, 3)) + eps);   % TCP cost

V = Z .* weight;
ideal_best  = max(V, [], 1);
ideal_worst = min(V, [], 1);
D_best  = sqrt(sum((V - ideal_best).^2, 2));
D_worst = sqrt(sum((V - ideal_worst).^2, 2));
score = D_worst ./ (D_best + D_worst + eps);

[score_sort, score_id] = sort(score, 'descend');
best3_id = score_id(1:3);
fprintf('Top-3 solutions:\n');
for r = 1:3
    idx = score_id(r);
    fprintf('  #%d  score=%.4f  PPR=%.3f  CWP=%.3f  TCP=%.1f  (%s)\n', r, score(idx), Y(idx,1), Y(idx,2), Y(idx,3), char(alg{idx}));
end

%% 3. Export tables (delete first)
files = {fullfile(out_dir,'TOPSIS_ranking.xlsx'), fullfile(out_dir,'TOPSIS_top3.xlsx'), fullfile(out_dir,'optimal_solution.xlsx')};
for i = 1:numel(files), if exist(files{i},'file'), delete(files{i}); end; end

header = [{'Rank','Algorithm'}, feature_names, {'PPR','CWP','TCP'}, {'Z_PPR','Z_CWP','Z_TCP'}, {'D_best','D_worst','Score'}];
data_cell = cell(n, numel(header));
for i = 1:n
    idx = score_id(i);
    data_cell(i, :) = [i, alg{idx}, num2cell(X(idx,:)), num2cell(Y(idx,:)), num2cell(Z(idx,:)), D_best(idx), D_worst(idx), score(idx)];
end
writecell([header; data_cell], files{1});
writecell([header; data_cell(1:3, :)], files{2});

best = score_id(1);
summary = [{'Item','Value'};
    {'Algorithm source', char(alg{best})};
    {'TOPSIS score', score(best)};
    [feature_names(:), num2cell(X(best,:)')];
    {'PPR (penetration rate, mm/rev)', Y(best,1)};
    {'CWP (cutterhead wear pressure)', Y(best,2)};
    {'TCP (cutterhead total power, kW)', Y(best,3)}];
writecell(summary, files{3});

%% 4. Figures
plot_topsis_bar(score_sort, out_dir);
plot_topsis_3d(Y, score, best3_id, out_dir);
plot_topsis_2d(Y, best3_id, out_dir);

fprintf('Step 6 completed.\n');

%% ==================== Local functions ====================
function plot_topsis_bar(score_sort, out_dir)
    font_name = 'Times New Roman';
    n = numel(score_sort);
    fig = figure('Position', [100 100 950 520], 'Color', 'w');
    ax = axes(fig);
    bar(ax, 1:n, score_sort, 0.9, 'FaceColor', [0.20 0.45 0.75], 'FaceAlpha', 0.85, 'EdgeColor', 'none');
    hold(ax, 'on');
    bar(ax, 1:3, score_sort(1:3), 0.9, 'FaceColor', [0.85 0.33 0.30], 'EdgeColor', 'none');
    xlabel(ax, 'Pareto solution rank', 'FontName', font_name, 'FontSize', 13);
    ylabel(ax, 'TOPSIS score', 'FontName', font_name, 'FontSize', 13);
    grid(ax, 'on'); ax.Box = 'on';
    set(ax, 'FontName', font_name, 'FontSize', 11, 'LineWidth', 1.0);
    export_figure(fig, fullfile(out_dir, 'Fig1_TOPSIS_ranking'));
    if isvalid(fig); close(fig); end
end

function plot_topsis_3d(Y, score, best3_id, out_dir)
    font_name = 'Times New Roman';
    fig = figure('Position', [100 100 900 720], 'Color', 'w');
    ax = axes(fig); hold(ax, 'on');
    scatter3(ax, Y(:,1), Y(:,2), Y(:,3), 30, score, 'filled', 'MarkerFaceAlpha', 0.7);
    colormap(ax, jet); cb = colorbar(ax); cb.Label.String = 'TOPSIS score';
    scatter3(ax, Y(best3_id,1), Y(best3_id,2), Y(best3_id,3), 220, [0.85 0.15 0.15], 'Marker', 'p', 'LineWidth', 1.5);
    xlabel(ax, 'PPR \uparrow', 'FontName', font_name, 'FontSize', 13);
    ylabel(ax, 'CWP \downarrow', 'FontName', font_name, 'FontSize', 13);
    zlabel(ax, 'TCP \downarrow', 'FontName', font_name, 'FontSize', 13);
    grid(ax, 'on'); ax.Box = 'on'; view(ax, [135 25]);
    set(ax, 'FontName', font_name, 'FontSize', 11, 'LineWidth', 1.0);
    export_figure(fig, fullfile(out_dir, 'Fig2_TOPSIS_pareto'));
    if isvalid(fig); close(fig); end
end

function plot_topsis_2d(Y, best3_id, out_dir)
    font_name = 'Times New Roman';
    pairs = {[1 2], [1 3], [2 3]};
    xlabs = {'PPR \uparrow', 'PPR \uparrow', 'CWP \downarrow'};
    ylabs = {'CWP \downarrow', 'TCP \downarrow', 'TCP \downarrow'};
    fig = figure('Position', [100 100 1200 400], 'Color', 'w');
    for s = 1:3
        subplot(1, 3, s);
        scatter(Y(:, pairs{s}(1)), Y(:, pairs{s}(2)), 25, [0.20 0.45 0.75], 'filled', 'MarkerFaceAlpha', 0.6);
        hold on;
        scatter(Y(best3_id, pairs{s}(1)), Y(best3_id, pairs{s}(2)), 120, [0.85 0.15 0.15], 'p', 'LineWidth', 1.5);
        xlabel(xlabs{s}, 'FontName', font_name, 'FontSize', 12);
        ylabel(ylabs{s}, 'FontName', font_name, 'FontSize', 12);
        grid on; box on;
        set(gca, 'FontName', font_name, 'FontSize', 11, 'LineWidth', 1.0);
    end
    export_figure(fig, fullfile(out_dir, 'Fig3_TOPSIS_2d'));
    if isvalid(fig); close(fig); end
end

function [outlier_idx, info] = detect_outliers_hybrid(X, target_idx, feature_idx, window_size, contamination, n_bins)
% DETECT_OUTLIERS_HYBRID  Hybrid sliding-window + binning + Isolation-Forest
%                        outlier detection (adapted from abnorm_detect.m,
%                        detect_label == 5).
%
%   [outlier_idx, info] = detect_outliers_hybrid(X, target_idx, feature_idx, ...
%                                                window_size, contamination, n_bins)
%
%   Inputs:
%       X             - data matrix [n_samples, n_variables]
%       target_idx    - column indices of the target variables (default: last 3)
%       feature_idx   - column indices of the feature variables
%                       (default: all columns not in target_idx)
%       window_size   - sliding-window length for the moving-median clip (default 15)
%       contamination - base outlier fraction for the Isolation Forest (default 0.03)
%       n_bins        - number of bins along the most-correlated feature (default 10)
%
%   Outputs:
%       outlier_idx   - logical vector, true for samples classified as outliers
%       info          - struct array (one element per target) carrying the
%                       feature/target data, fitted curve and per-target outlier
%                       flags, used by the plotting routine.
%
%   Method (three stages):
%     1) Sliding window: clip each variable with filloutliers(...,"clip",
%        "movmedian",window) so extreme values are bounded by their local
%        neighbourhood -- this avoids leakage from global statistics.
%     2) For each target: pick the feature most correlated with it, fit a
%        quadratic centre curve target = f(feature), bin the samples along that
%        feature, and run an Isolation Forest on the target values inside each
%        bin (with a bin-size-adaptive contamination fraction).
%     3) Union the per-target, per-bin outliers; a sample is removed if any
%        target flags it.

    n = size(X, 1);
    p = size(X, 2);

    if nargin < 2 || isempty(target_idx)
        target_idx = p-2:p;                          % last 3 columns
    end
    if nargin < 3 || isempty(feature_idx)
        feature_idx = setdiff(1:p, target_idx);
    end
    if nargin < 4 || isempty(window_size),   window_size   = 15;  end
    if nargin < 5 || isempty(contamination), contamination = 0.03; end
    if nargin < 6 || isempty(n_bins),        n_bins        = 10;  end

    %% ---- Stage 1: sliding-window clipping (moving median) ----
    X_clip = X;
    for j = 1:p
        X_clip(:, j) = filloutliers(X(:, j), 'clip', 'movmedian', window_size);
    end

    %% ---- Stage 2: per-target binning + Isolation Forest ----
    Xz = zscore(X_clip);                    % standardize for correlation / binning

    outlier_union = false(n, 1);
    info = struct([]);

    for t = 1:numel(target_idx)
        tcol = target_idx(t);
        yz   = Xz(:, tcol);                 % standardized target

        % (a) Feature most correlated with this target.
        corrs = zeros(numel(feature_idx), 1);
        for k = 1:numel(feature_idx)
            cc = corrcoef(Xz(:, feature_idx(k)), yz);
            if ~isnan(cc(1,2)), corrs(k) = abs(cc(1,2)); end
        end
        [~, kmax] = max(corrs);
        fcol = feature_idx(kmax);
        xz   = Xz(:, fcol);                 % standardized driving feature

        % (b) 3-sigma pre-filter on the driving feature (standardized).
        keep3 = abs(xz) <= 3;

        % (c) Bin along the driving feature; run Isolation Forest per bin.
        edges = linspace(min(xz(keep3)), max(xz(keep3)), n_bins + 1);
        bin_outlier = false(n, 1);
        for b = 1:n_bins
            if b < n_bins
                idx = find(keep3 & xz >= edges(b) & xz <  edges(b+1));
            else
                idx = find(keep3 & xz >= edges(b) & xz <= edges(b+1));
            end
            if numel(idx) < 20, continue; end

            % Bin-size-adaptive contamination fraction. Oversized bins get a
            % REDUCED rate (matching the original abnorm_detect logic), so the
            % total flagged fraction stays near `contamination`.
            bin_fraction = numel(idx) / sum(keep3);
            expected = 1 / n_bins;
            if bin_fraction > expected
                rate = contamination / (bin_fraction / expected);
            else
                rate = contamination;
            end
            rate = max(min(rate, 0.5), 0.005);

            [~, tf] = iforest(yz(idx), 'ContaminationFraction', rate);
            bin_outlier(idx(tf)) = true;
        end

        % (e) Final per-target decision: 3-sigma feature outliers + bin outliers.
        outlier_this = bin_outlier | ~keep3;
        outlier_union = outlier_union | outlier_this;

        % Store plotting data in the ORIGINAL (unstandardized) space.
        info(t).target   = tcol;
        info(t).feature  = fcol;
        info(t).xf       = X_clip(:, fcol);
        info(t).y        = X_clip(:, tcol);
        info(t).outlier  = outlier_this;
        info(t).fitFcn   = polyfit(X_clip(keep3, fcol), X_clip(keep3, tcol), 2);
        info(t).xgrid    = linspace(min(X_clip(:, fcol)), max(X_clip(:, fcol)), 200);
        info(t).ygrid    = polyval(info(t).fitFcn, info(t).xgrid);
    end

    outlier_idx = outlier_union;
end

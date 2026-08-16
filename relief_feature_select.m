function [weights_avg, weights_each] = relief_feature_select(X_features, Y_targets, k)
% RELIEF_FEATURE_SELECT  Multi-target Relief (RReliefF) importance.
% For each target, relieff is run separately; each target's importance is
% clipped at zero and normalised to unit sum, then the three vectors are
% averaged to obtain a single multi-output importance per feature.

    if nargin < 3 || isempty(k), k = 10; end

    n_targets = size(Y_targets, 2);
    n_feat    = size(X_features, 2);

    % Standardise features: relieff is distance-based, so raw features with
    % very different scales (e.g. total thrust in kN vs. grouting pressure in
    % MPa) would otherwise dominate the nearest-neighbour search.
    Xz = zscore(X_features);

    weights_each = zeros(n_targets, n_feat);
    for t = 1:n_targets
        [~, w] = relieff(Xz, Y_targets(:, t), k);
        w = max(w, 0);          % clip negative (irrelevant) weights
        if sum(w) > 0
            w = w / sum(w);     % unit-sum normalisation (remove scale)
        end
        weights_each(t, :) = w;
    end

    weights_avg = mean(weights_each, 1);   % average over the three targets
end

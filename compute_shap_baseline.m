function [shap_values, X_shap, sample_idx] = compute_shap_baseline(models, X_test_f, X_train_f, num_sample, rng_seed)
% COMPUTE_SHAP_BASELINE  Approximate SHAP values via single-feature ablation
% against the training-set mean (following the original code). Operates on
% raw (un-normalised) data, so contributions are in the target's own units.
% The ablation is vectorised over samples for speed.
%
%   [shap_values, X_shap, sample_idx] = ...
%       compute_shap_baseline(models, X_test_f, X_train_f, num_sample, rng_seed)
%
%   Inputs:
%       models    - 1xK cell array of TreeBagger regression models
%       X_test_f  - test-set features (N x M)
%       X_train_f - train-set features (used for the background mean)
%       num_sample- number of test samples to explain (default 200)
%       rng_seed  - random seed (default 42)
%   Outputs:
%       shap_values - num_sample x M x K matrix of SHAP contributions
%       X_shap      - num_sample x M explained feature values
%       sample_idx  - indices of the explained samples

    if nargin < 4 || isempty(num_sample), num_sample = 200; end
    if nargin < 5 || isempty(rng_seed),  rng_seed  = 42;  end

    rng(rng_seed);
    num_test = size(X_test_f, 1);
    num_sample = min(num_sample, num_test);
    sample_idx = randperm(num_test, num_sample);
    X_shap = X_test_f(sample_idx, :);

    x_base = mean(X_train_f, 1);   % background mean per feature
    num_feat = size(X_test_f, 2);
    num_target = numel(models);

    shap_values = zeros(num_sample, num_feat, num_target);

    for k = 1:num_target
        mdl = models{k};
        y_full = predict(mdl, X_shap);           % num_sample x 1
        for j = 1:num_feat
            X_replace = X_shap;
            X_replace(:, j) = x_base(j);
            y_replace = predict(mdl, X_replace); % num_sample x 1
            shap_values(:, j, k) = y_full - y_replace;
        end
    end
    shap_values(~isfinite(shap_values)) = 0;
end

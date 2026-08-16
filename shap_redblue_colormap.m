function cmap = shap_redblue_colormap(n)
% SHAP_REDBLUE_COLORMAP  Red-blue diverging colormap (blue = low feature
% value, red = high feature value), matching the standard SHAP summary-plot
% colour scheme.
%
%   cmap = shap_redblue_colormap(n)  returns an n-by-3 RGB colormap.

    if nargin < 1 || isempty(n), n = 256; end
    half = floor(n / 2);
    % blue -> white (low half)
    r1 = linspace(0.23, 1, half)';
    g1 = linspace(0.30, 1, half)';
    b1 = linspace(0.75, 1, half)';
    % white -> red (high half)
    r2 = linspace(1, 0.71, n - half)';
    g2 = linspace(1, 0.13, n - half)';
    b2 = linspace(1, 0.13, n - half)';
    cmap = [r1 g1 b1; r2 g2 b2];
end

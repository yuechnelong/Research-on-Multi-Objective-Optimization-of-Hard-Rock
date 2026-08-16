function export_figure(fig, filepath)
% EXPORT_FIGURE  Export a figure in PNG, EMF and FIG formats.
%
%   export_figure(fig, filepath)
%
%   Writes:
%       filepath.png  - high-resolution raster (600 DPI)
%       filepath.emf  - vector image (Enhanced Metafile, for Word documents)
%       filepath.fig  - editable MATLAB figure
%
%   Export order matters in headless (-batch) mode: FIG and EMF are written
%   first (they keep the handle fresh), and the raster PNG is written last.
%   Each step is guarded so a failure never aborts the remaining exports.

    drawnow;   % flush the figure before exporting

    % FIG (editable) first.
    try
        savefig(fig, [filepath '.fig']);
    catch err
        warning('FIG export failed for %s: %s', filepath, err.message);
    end

    % EMF (vector) second.
    try
        saveas(fig, [filepath '.emf'], 'emf');
    catch err
        warning('EMF export failed for %s: %s', filepath, err.message);
    end

    % PNG (high-resolution raster) last. `print` is used instead of
    % exportgraphics for stability in headless (-batch) mode, where
    % exportgraphics can invalidate the figure handle.
    try
        print(fig, [filepath '.png'], '-dpng', '-r600');
    catch err
        warning('PNG export failed for %s: %s', filepath, err.message);
    end
end

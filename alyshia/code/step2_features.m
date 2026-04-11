%% Step 2: Build Feature Matrix
% Extracts 459 features per window using
% the unified extract_all_features function.
% Outputs: X, y, win_info, feature_names

addpath('utils');
fprintf('=== STEP 2: Build Feature Matrix ===\n');

window_sec = 2;
overlap_pct = 50;
window_len = window_sec * Fs;
hop_len = floor(window_len * (1 - overlap_pct/100));

cache_file = 'feature_cache.mat';
if exist(cache_file, 'file')
    fprintf('Loading from cache...\n');
    load(cache_file, 'X', 'y', 'win_info', 'feature_names');
    fprintf('  %d windows x %d features\n', size(X,1), size(X,2));
    fprintf('Step 2 done (cached).\n\n');
    return;
end

perturbation_types = {'Slip','ITD','ITR','Hit','LOS','Trip','Coll','Miss'};
X = []; y = []; win_info = {};
participants = fieldnames(filtered_data);

for p = 1:length(participants)
    pID = participants{p};
    trials = fieldnames(filtered_data.(pID));
    for t = 1:length(trials)
        tID = trials{t};
        trial = filtered_data.(pID).(tID);
        if ~isfield(trial,'Back') || isempty(trial.Back), continue; end

        n_samples = size(trial.Back, 1);
        if n_samples < window_len, continue; end

        % Find perturbation type and event windows
        matched_pert = '';
        for pt = 1:length(perturbation_types)
            if contains(tID, perturbation_types{pt})
                matched_pert = perturbation_types{pt}; break;
            end
        end

        fall_windows = []; nearfall_windows = [];
        if ~isempty(matched_pert) && isfield(segments,pID) && isfield(segments.(pID),matched_pert)
            seg = segments.(pID).(matched_pert);
            for fi = 1:length(seg.falls)
                ev = seg.falls(fi);
                e_end = ev.end_time;
                if isnan(e_end), e_end = ev.fall_time + 5*Fs; end
                fall_windows = [fall_windows; ev.movement_start, e_end];
            end
            for ni = 1:length(seg.nearfalls)
                ev = seg.nearfalls(ni);
                if isnan(ev.nearfall_end), continue; end
                nearfall_windows = [nearfall_windows; ev.movement_start, ev.nearfall_end];
            end
        end

        n_win = floor((n_samples - window_len) / hop_len) + 1;
        warnState = warning('off','signal:findpeaks:largeMinPeakHeight');
        for w = 1:n_win
            w_start = (w-1)*hop_len + 1;
            w_end = w_start + window_len - 1;

            % Label: 2=fall (priority), 1=near-fall, 0=ADL
            label = 0;
            for fi = 1:size(fall_windows,1)
                ov = min(w_end,fall_windows(fi,2)) - max(w_start,fall_windows(fi,1));
                if ov > 0 && ov/window_len > 0.5, label = 2; break; end
            end
            if label == 0
                for ni = 1:size(nearfall_windows,1)
                    ov = min(w_end,nearfall_windows(ni,2)) - max(w_start,nearfall_windows(ni,1));
                    if ov > 0 && ov/window_len > 0.5, label = 1; break; end
                end
            end

            features = extract_all_features(trial, w_start, w_end, Fs);
            X = [X; features];
            y = [y; label];
            win_info{end+1} = sprintf('%s/%s/%d-%d', pID, tID, w_start, w_end);
        end
        warning(warnState);
    end
    fprintf('  %s: %d windows\n', pID, length(y));
end

X(isnan(X))=0; X(isinf(X))=0;
[~, feature_names] = extract_all_features(filtered_data.(participants{1}).(fieldnames(filtered_data.(participants{1})){1}), 1, window_len, Fs);

fprintf('  %d windows x %d features\n', size(X,1), size(X,2));
fprintf('  ADL=%d, NearFall=%d, Fall=%d\n', sum(y==0), sum(y==1), sum(y==2));
save(cache_file, 'X', 'y', 'win_info', 'feature_names', '-v7.3');
fprintf('Step 2 done.\n\n');

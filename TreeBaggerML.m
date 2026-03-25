% TreeBagger training with location-robust IMU features (magnitude + aggregation)
% + SAFE filtering + SAFE feature extraction (prevents filtfilt + FFT edge-case errors)

clear; clc; close all;
%% Load Data (loads all_data, clean_labels)
load("GrandChallengeData.mat");

%% Parameters
Fs = 1000;                 % keep same default (dataset timestamps are in seconds)
window_size_sec = 3;       % teammate used 3s windows
expected_length = [];      % learned after first good feature vector

% Include both training + test IMU names (robust to placement differences)
imu_sources = {'Back','Right_Thigh','Left_Thigh','Sternum','Left_Arm','Right_Arm'};
adl_keywords = {'Stand','Stairs','Jog','Pick','Sit','Walk','Lie','JJ'};

X = [];
y = [];

%% SAFE FILTER HELPERS
function yout = safe_filtfilt(b,a,xin)
    % filtfilt needs sufficient length; if too short, return raw
    if size(xin,1) <= 24
        yout = xin;
        return;
    end
    yout = filtfilt(b,a,xin);
end

function yout = lpf_imu(xin, Fs)
    [b,a] = butter(4, 10/(Fs/2), 'low');
    yout = safe_filtfilt(b,a,xin);
end

function yout = bpf_ecg(xin, Fs)
    [b,a] = butter(4, [0.5 40]/(Fs/2), 'bandpass');
    yout = safe_filtfilt(b,a,xin);
end

function yout = lpf_gss(xin, Fs)
    [b,a] = butter(4, 5/(Fs/2), 'low');
    yout = safe_filtfilt(b,a,xin);
end

%% SAFE FEATURE EXTRACTION WRAPPER
function fv = safe_feature_extract(sig, Fs, minN)
    % Returns [] if signal too short or invalid; otherwise feature_extract_591k
    if nargin < 3, minN = 50; end
    if isempty(sig) || numel(sig) < minN || any(~isfinite(sig)) || std(sig) == 0
        fv = [];
        return;
    end
    fv = feature_extract_591k(sig, Fs);
end

%% LOCATION-ROBUST WINDOW FEATURE BUILDER
function fv = extract_features_from_window(data_struct, win_start, win_end, Fs, imu_sources)
    fv = [];

    % IMU magnitude features per sensor
    accFeatAll = [];
    gyrFeatAll = [];

    for s = 1:length(imu_sources)
        src = imu_sources{s};
        if ~isfield(data_struct, src), continue; end

        imu = data_struct.(src);
        idx = imu(:,1) >= win_start & imu(:,1) <= win_end;
        window = imu(idx,:);

        % Need [t ax ay az gx gy gz]
        if size(window,2) < 7 || size(window,1) < 10
            continue;
        end

        acc = window(:,2:4);
        gyr = window(:,5:7);

        accF = lpf_imu(acc, Fs);
        gyrF = lpf_imu(gyr, Fs);

        acc_mag = sqrt(sum(accF.^2,2));
        gyr_mag = sqrt(sum(gyrF.^2,2));

        f_acc = safe_feature_extract(acc_mag, Fs, 50);
        f_gyr = safe_feature_extract(gyr_mag, Fs, 50);

        if ~isempty(f_acc), accFeatAll(end+1,:) = f_acc; end %#ok<AGROW>
        if ~isempty(f_gyr), gyrFeatAll(end+1,:) = f_gyr; end %#ok<AGROW>
    end

    % Require at least one IMU-derived feature set; otherwise skip window
    if isempty(accFeatAll) && isempty(gyrFeatAll)
        fv = [];
        return;
    end

    % Aggregate across IMUs (placement-robust)
    if ~isempty(accFeatAll)
        fv = [fv, mean(accFeatAll,1), max(accFeatAll,[],1)];
    end
    if ~isempty(gyrFeatAll)
        fv = [fv, mean(gyrFeatAll,1), max(gyrFeatAll,[],1)];
    end

    % ECG 
    if isfield(data_struct,'ECG')
        ecg = data_struct.ECG;
        idx = ecg(:,1) >= win_start & ecg(:,1) <= win_end;
        sig = ecg(idx,2);

        if numel(sig) >= 50 && all(isfinite(sig)) && std(sig) > 0
            sig = bpf_ecg(sig, Fs);
            f_ecg = safe_feature_extract(sig, Fs, 50);
            if ~isempty(f_ecg), fv = [fv, f_ecg]; end
        end
    end

    % GSS 
    if isfield(data_struct,'GSS')
        gss = data_struct.GSS;
        idx = gss(:,1) >= win_start & gss(:,1) <= win_end;
        sig = gss(idx,2);

        if numel(sig) >= 50 && all(isfinite(sig)) && std(sig) > 0
            sig = lpf_gss(sig, Fs);
            f_gss = safe_feature_extract(sig, Fs, 50);
            if ~isempty(f_gss), fv = [fv, f_gss]; end
        end
    end
end

%% EVENT-BASED WINDOWS (Falls / Near-Falls)
participants = fieldnames(clean_labels);

for p = 1:length(participants)
    participant = participants{p};
    if ~isfield(all_data, participant), continue; end

    trials = fieldnames(clean_labels.(participant));

    for t = 1:length(trials)
        trial = trials{t};

        % Match trial name in all_data.(participant)
        data_trials = fieldnames(all_data.(participant));
        match_idx = find(contains(data_trials, trial, 'IgnoreCase', true), 1);
        if isempty(match_idx), continue; end
        data_trial = data_trials{match_idx};

        labels = clean_labels.(participant).(trial);
        if isempty(labels), continue; end

        % NOTE: keeping teammate grouping to preserve structure
        group_id = [1; cumsum(diff(labels(:,3))~=0)+1];
        groups = unique(group_id);

        for g = groups'
            idx = group_id == g;
            grp_labels = labels(idx,1);
            grp_times  = labels(idx,2);

            if any(grp_labels == 4)
                event_time = mean(grp_times(grp_labels==4))/1000;
                label = 2;  % fall
            elseif any(grp_labels == 2)
                event_time = mean(grp_times(grp_labels==2))/1000;
                label = 1;  % near-fall
            else
                continue;
            end

            win_start = max(0, event_time - window_size_sec/2);
            win_end   = win_start + window_size_sec;

            fv = extract_features_from_window( ...
                all_data.(participant).(data_trial), win_start, win_end, Fs, imu_sources);

            if ~isempty(fv)
                if isempty(expected_length)
                    expected_length = length(fv);
                end
                if length(fv) == expected_length
                    X = [X; fv];
                    y = [y; label];
                end
            end
        end
    end
end

%% ADL WINDOWS
adl_stride = 1.5;
num_adl_windows = 3;

for p = 1:length(participants)
    participant = participants{p};
    if ~isfield(all_data, participant), continue; end

    trials = fieldnames(all_data.(participant));

    for t = 1:length(trials)
        trial = trials{t};

        % Only use trials that look like ADL based on keywords
        if ~any(contains(trial, adl_keywords)), continue; end

        data_struct = all_data.(participant).(trial);

        % Pick any available IMU as time base (do NOT require Back)
        imuAvail = imu_sources(ismember(imu_sources, fieldnames(data_struct)));
        if isempty(imuAvail), continue; end

        imu0 = data_struct.(imuAvail{1});
        if isempty(imu0) || size(imu0,2) < 2
            continue;
        end

        tvec = imu0(:,1);

        % Guard against empty/short signals
        if isempty(tvec) || numel(tvec) < 2
            continue;
        end

        % Guard: must be long enough for a full window
        if (tvec(end) - tvec(1)) < window_size_sec
            continue;
        end

        % Candidate start times for windows
        start_times = tvec(1):adl_stride:(tvec(end)-window_size_sec);
        if isempty(start_times), continue; end

        nStarts = length(start_times);
        mid = round(nStarts/2);

        % Choose a few windows spread across the trial (early/mid/late)
        idxs = unique([1:min(3,nStarts), mid-1:mid+1, max(nStarts-2,1):nStarts]);
        idxs = idxs(idxs>=1 & idxs<=nStarts);

        for i = 1:min(num_adl_windows, length(idxs))
            win_start = start_times(idxs(i));
            win_end   = win_start + window_size_sec;

            fv = extract_features_from_window(data_struct, win_start, win_end, Fs, imu_sources);

            if ~isempty(fv)
                if isempty(expected_length)
                    expected_length = length(fv);
                end
                if length(fv) == expected_length
                    X = [X; fv];
                    y = [y; 0];  % ADL
                end
            end
        end
    end
end

%% Normalize (only if we have enough samples)
if isempty(X)
    error("No training samples collected. Check windowing/fieldnames.");
end

X = zscore(X);

%% Summary
fprintf('\nTraining summary:\n');
fprintf('  ADL (0):       %d\n', sum(y==0));
fprintf('  Near-Fall (1): %d\n', sum(y==1));
fprintf('  Fall (2):      %d\n', sum(y==2));
fprintf('  Total:         %d\n', length(y));
fprintf('  Feature length: %d\n', size(X,2));

%% Train Model + save
Mdl = TreeBagger(100, X, string(y), ...
    'Method','classification', ...
    'OOBPrediction','on', ...
    'ClassNames',["0","1","2"], ...
    'Prior','empirical');

save('trained_model.mat','Mdl');
disp("Saved trained_model.mat");

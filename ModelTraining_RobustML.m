%% ModelTraining_RobustML.m
% TreeBagger training with location-robust IMU features (magnitude + aggregation)

clear; clc; close all;

%% Load Data
load("GrandChallengeData.mat");   % loads all_data, clean_labels

%% Parameters
Fs = 1000;                 % if your time vectors are seconds, this is fine)
window_size_sec = 3;       % used 3 s windows
expected_length = [];      % learn after first successful fv

% Include both training and test IMU fieldnames (robust to placement)
imu_sources = {'Back','Right_Thigh','Left_Thigh','Sternum','Left_Arm','Right_Arm'};
adl_keywords = {'Stand','Stairs','Jog','Pick','Sit','Walk','Lie','JJ'};

X = [];
y = [];

%% Filters
function yout = lpf_imu(xin, Fs)
    [b,a] = butter(4, 10/(Fs/2), 'low');
    yout = filtfilt(b,a,xin);
end

function yout = bpf_ecg(xin, Fs)
    [b,a] = butter(4, [0.5 40]/(Fs/2), 'bandpass');
    yout = filtfilt(b,a,xin);
end

function yout = lpf_gss(xin, Fs)
    [b,a] = butter(4, 5/(Fs/2), 'low');
    yout = filtfilt(b,a,xin);
end

%% Feature Extraction Helper (LOCATION-ROBUST)
function fv = extract_features_from_window(data_struct, win_start, win_end, Fs, imu_sources)
    fv = [];

    % IMU: magnitude features per sensor -> aggregate across sensors 
    accFeatAll = [];
    gyrFeatAll = [];

    for s = 1:length(imu_sources)
        src = imu_sources{s};
        if ~isfield(data_struct, src)
            continue;
        end

        imu = data_struct.(src);
        idx = imu(:,1) >= win_start & imu(:,1) <= win_end;
        window = imu(idx,:);

        % Expect columns: [t ax ay az gx gy gz]
        if size(window,2) < 7 || size(window,1) < 10
            continue;
        end

        acc = window(:,2:4);
        gyr = window(:,5:7);

        % Filter (10 Hz LP) on both (consistent with earlier settings)
        accF = lpf_imu(acc, Fs);
        gyrF = lpf_imu(gyr, Fs);

        acc_mag = sqrt(sum(accF.^2,2));
        gyr_mag = sqrt(sum(gyrF.^2,2));

        if ~isempty(acc_mag) && all(isfinite(acc_mag))
            accFeatAll(end+1,:) = feature_extract_591k(acc_mag, Fs); %#ok<AGROW>
        end
        if ~isempty(gyr_mag) && all(isfinite(gyr_mag))
            gyrFeatAll(end+1,:) = feature_extract_591k(gyr_mag, Fs); %#ok<AGROW>
        end
    end

    % Create stable dummy vector length if no IMU found in this trial
    dummy = feature_extract_591k(zeros(200,1), Fs);

    if isempty(accFeatAll)
        fv = [fv, zeros(1,length(dummy)), zeros(1,length(dummy))]; % mean + max placeholders
    else
        fv = [fv, mean(accFeatAll,1), max(accFeatAll,[],1)];
    end

    if isempty(gyrFeatAll)
        fv = [fv, zeros(1,length(dummy)), zeros(1,length(dummy))];
    else
        fv = [fv, mean(gyrFeatAll,1), max(gyrFeatAll,[],1)];
    end

    % ECG
    if isfield(data_struct,'ECG')
        ecg = data_struct.ECG;
        idx = ecg(:,1) >= win_start & ecg(:,1) <= win_end;
        sig = ecg(idx,2);
        if ~isempty(sig) && all(~isnan(sig))
            sig = bpf_ecg(sig, Fs);
            fv = [fv, feature_extract_591k(sig, Fs)];
        else
            fv = [fv, dummy];
        end
    else
        fv = [fv, dummy];
    end

    % GSS
    if isfield(data_struct,'GSS')
        gss = data_struct.GSS;
        idx = gss(:,1) >= win_start & gss(:,1) <= win_end;
        sig = gss(idx,2);
        if ~isempty(sig) && all(~isnan(sig))
            sig = lpf_gss(sig, Fs);
            fv = [fv, feature_extract_591k(sig, Fs)];
        else
            fv = [fv, dummy];
        end
    else
        fv = [fv, dummy];
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

        % NOTE: labels(:,3) may not be ideal.
        group_id = [1; cumsum(diff(labels(:,3))~=0)+1];
        groups = unique(group_id);

        for g = groups'
            idx = group_id == g;
            grp_labels = labels(idx,1);
            grp_times = labels(idx,2);

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
            win_end = win_start + window_size_sec;

            fv = extract_features_from_window( ...
                all_data.(participant).(data_trial), ...
                win_start, win_end, Fs, imu_sources);

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
        if ~any(contains(trial, adl_keywords)), continue; end

        data_struct = all_data.(participant).(trial);

        % Choose any available IMU as time base
        imuAvail = imu_sources(ismember(imu_sources, fieldnames(data_struct)));
        if isempty(imuAvail), continue; end
        tvec = data_struct.(imuAvail{1})(:,1);

        start_times = tvec(1):adl_stride:(tvec(end)-window_size_sec);
        if isempty(start_times), continue; end

        mid = round(length(start_times)/2);
        idxs = unique([1:min(3,end), mid-1:mid+1, max(end-2,1):end]);
        idxs = idxs(idxs>=1 & idxs<=length(start_times));

        for i = 1:min(num_adl_windows,length(idxs))
            win_start = start_times(idxs(i));
            win_end = win_start + window_size_sec;

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

%% Normalize
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

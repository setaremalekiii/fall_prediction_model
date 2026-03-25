%% Load Data
load("GrandChallengeData.mat");

%% Parameters
Fs = 1000;
window_size_sec = 3;
expected_length = 280;

imu_sources = {'Back', 'Right_Thigh', 'Left_Thigh'};
adl_keywords = {'Stand','Stairs','Jog','Pick','Sit','Walk','Lie','JJ'};

X = [];
y = [];

%% Filters
function y = lpf_imu(x, Fs)
    [b,a] = butter(4, 10/(Fs/2), 'low');
    y = filtfilt(b,a,x);
end

function y = bpf_ecg(x, Fs)
    [b,a] = butter(4, [0.5 40]/(Fs/2), 'bandpass');
    y = filtfilt(b,a,x);
end

function y = lpf_gss(x, Fs)
    [b,a] = butter(4, 5/(Fs/2), 'low');
    y = filtfilt(b,a,x);
end

%% Feature Extraction Helper
function fv = extract_features_from_window(data_struct, win_start, win_end, Fs, imu_sources)
    fv = [];

    % IMU
    for s = 1:length(imu_sources)
        src = imu_sources{s};
        if isfield(data_struct, src)
            imu = data_struct.(src);
            idx = imu(:,1) >= win_start & imu(:,1) <= win_end;
            window = imu(idx,:);
            
            for ch = 2:min(7, size(window,2))
                sig = window(:,ch);
                if ~isempty(sig) && all(~isnan(sig))
                    sig = lpf_imu(sig, Fs);
                    fv = [fv, feature_extract_591k(sig, Fs)];
                end
            end
        end
    end

    % ECG
    if isfield(data_struct,'ECG')
        ecg = data_struct.ECG;
        idx = ecg(:,1) >= win_start & ecg(:,1) <= win_end;
        sig = ecg(idx,2);
        if ~isempty(sig) && all(~isnan(sig))
            sig = bpf_ecg(sig, Fs);
            fv = [fv, feature_extract_591k(sig, Fs)];
        end
    end

    % GSS
    if isfield(data_struct,'GSS')
        gss = data_struct.GSS;
        idx = gss(:,1) >= win_start & gss(:,1) <= win_end;
        sig = gss(idx,2);
        if ~isempty(sig) && all(~isnan(sig))
            sig = lpf_gss(sig, Fs);
            fv = [fv, feature_extract_591k(sig, Fs)];
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

        % Match trial name
        data_trials = fieldnames(all_data.(participant));
        match_idx = find(contains(data_trials, trial, 'IgnoreCase', true),1);
        if isempty(match_idx), continue; end
        data_trial = data_trials{match_idx};

        labels = clean_labels.(participant).(trial);
        if isempty(labels), continue; end

        % Group events
        group_id = [1; cumsum(diff(labels(:,3))~=0)+1];
        groups = unique(group_id);

        for g = groups'
            idx = group_id == g;
            grp_labels = labels(idx,1);
            grp_times = labels(idx,2);

            if any(grp_labels == 4)
                event_time = mean(grp_times(grp_labels==4))/1000;
                label = 2;
            elseif any(grp_labels == 2)
                event_time = mean(grp_times(grp_labels==2))/1000;
                label = 1;
            else
                continue;
            end

            win_start = max(0, event_time - window_size_sec/2);
            win_end = win_start + window_size_sec;

            fv = extract_features_from_window( ...
                all_data.(participant).(data_trial), ...
                win_start, win_end, Fs, imu_sources);

            if ~isempty(fv) && abs(length(fv)-expected_length)<=10
                X = [X; fv];
                y = [y; label];
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

        if ~isfield(data_struct,'Back'), continue; end
        tvec = data_struct.Back(:,1);

        start_times = tvec(1):adl_stride:(tvec(end)-window_size_sec);
        if isempty(start_times), continue; end

        mid = round(length(start_times)/2);
        idxs = unique([1:min(3,end), mid-1:mid+1, max(end-2,1):end]);
        idxs = idxs(idxs>=1 & idxs<=length(start_times));

        for i = 1:min(num_adl_windows,length(idxs))
            win_start = start_times(idxs(i));
            win_end = win_start + window_size_sec;

            fv = extract_features_from_window(data_struct, win_start, win_end, Fs, imu_sources);

            if ~isempty(fv) && abs(length(fv)-expected_length)<=10
                X = [X; fv];
                y = [y; 0];
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

%% Train Model + save
Mdl = TreeBagger(100, X, string(y), ...
    'Method','classification', ...
    'OOBPrediction','on', ...
    'ClassNames',["0","1","2"], ...
    'Prior','empirical');

save('trained_model.mat','Mdl');

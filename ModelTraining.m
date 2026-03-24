%% Loading training data
load("GrandChallengeData.mat");

%% Training model
Fs = 1000;
% Low-pass filter for IMU (Accel/Gyro)
function x_filtered = lowpass_filter_imu(x, Fs)
    if length(x) < 25
        x_filtered = x;  % skip filtering if too short
        return;
    end
    fc = 20;  % cutoff frequency
    [b, a] = butter(4, fc/(Fs/2), 'low');
    x_filtered = filtfilt(b, a, x);
end

function x_filtered = bandpass_filter_ecg(x, Fs)
    if length(x) < 25
        x_filtered = x;  % skip filtering if too short
        return;
    end
    [b, a] = butter(4, [0.5, 40]/(Fs/2), 'bandpass');
    x_filtered = filtfilt(b, a, x);
end
window_size_sec = 3;
participants = fieldnames(clean_labels);
expected_length = 280;
X = [];  % Feature matrix
y = [];  % Labels

for p = 1:length(participants)
    participant = participants{p};
    if ~isfield(all_data, participant), continue; end
    trials = fieldnames(clean_labels.(participant));
    
    for t = 1:length(trials)
        trial = trials{t};
        all_trial_names = fieldnames(all_data.(participant));
        matching_trial = '';
        for dt = 1:length(all_trial_names)
            if contains(all_trial_names{dt}, trial, 'IgnoreCase', true)
                matching_trial = all_trial_names{dt};
                break;
            end
        end
        if isempty(matching_trial), continue; end
        labels = clean_labels.(participant).(trial);
        if isempty(labels), continue; end
        
        group_id = ones(size(labels,1),1);
        if size(labels,1) > 1
            group_id(2:end) = cumsum(diff(labels(:,3))~=0) + 1;
        end
        unique_groups = unique(group_id);
        
        for g = 1:length(unique_groups)
            grp_idx = find(group_id == unique_groups(g));
            grp_labels = labels(grp_idx,1);
            grp_times = labels(grp_idx,2);
            
            if any(grp_labels == 4)
                event_time_ms = mean(grp_times(grp_labels == 4));
                mapped_label = 2;
            elseif any(grp_labels == 2)
                event_time_ms = mean(grp_times(grp_labels == 2));
                mapped_label = 1;
            else
                continue;
            end
            
            event_time = event_time_ms / 1000;
            win_start = max(0, event_time - window_size_sec/2);
            win_end = win_start + window_size_sec;
            
            feature_vector = [];
            imu_sources = {'Back', 'Right_Thigh', 'Left_Thigh', 'Sternum'};
            for s = 1:length(imu_sources)
                imu_name = imu_sources{s};
                if isfield(all_data.(participant).(matching_trial), imu_name)
                    imu = all_data.(participant).(matching_trial).(imu_name);
                    idx = imu(:,1) >= win_start & imu(:,1) <= win_end;
                    window = imu(idx,:);
                    for ch = 2:7
                        if size(window,2) >= ch
                            data = window(:,ch);
                            if ~isempty(data) && all(~isnan(data))
                                data = lowpass_filter_imu(data, Fs);
                                feature_vector = [feature_vector, feature_extract_591k(data, Fs)];
                            end
                        end
                    end
                end
            end
            % ECG
            if isfield(all_data.(participant).(matching_trial), 'ECG')
                ecg = all_data.(participant).(matching_trial).ECG;
                idx = ecg(:,1) >= win_start & ecg(:,1) <= win_end;
                window = ecg(idx,:);
                if size(window,2) >= 2
                    data = window(:,2);
                    if ~isempty(data) && all(~isnan(data))
                        data = bandpass_filter_ecg(data, Fs);
                        feature_vector = [feature_vector, feature_extract_591k(data, Fs)];
                    end
                end
            end
            % GSS
            if isfield(all_data.(participant).(matching_trial), 'GSS')
                gss = all_data.(participant).(matching_trial).GSS;
                idx = gss(:,1) >= win_start & gss(:,1) <= win_end;
                window = gss(idx,:);
                if size(window,2) >= 2
                    data = window(:,2);
                    if ~isempty(data) && all(~isnan(data))
                        data = bandpass_filter_ecg(data, Fs);
                        feature_vector = [feature_vector, feature_extract_591k(data, Fs)];
                    end
                end
            end
            if ~isempty(feature_vector) && abs(length(feature_vector) - expected_length) <= 10
                X = [X; feature_vector];
                y = [y; mapped_label];  % or 0 for ADL
            end
        end
    end
end

% ADL trial extraction
adl_keywords = {'Stand', 'Stairs', 'Jog', 'Pick', 'Sit', 'Walk', 'Lie', 'JJ'};
adl_window_stride = 1.5;
num_adl_windows = 3;

for p = 1:length(participants)
    participant = participants{p};
    if ~isfield(all_data, participant), continue; end
    trial_names = fieldnames(all_data.(participant));
    
    for t = 1:length(trial_names)
        trial = trial_names{t};
        is_adl = any(cellfun(@(k) contains(trial, k), adl_keywords));
        if ~is_adl, continue; end

        imu_fields = {'Back', 'Sternum', 'Right_Thigh', 'Left_Thigh'};
        imu_data = [];
        for f = 1:length(imu_fields)
            if isfield(all_data.(participant).(trial), imu_fields{f})
                imu_data = all_data.(participant).(trial).(imu_fields{f});
                break;
            end
        end
        if isempty(imu_data), continue; end

        duration = imu_data(end,1) - imu_data(1,1);
        step = adl_window_stride;
        win_len = window_size_sec;
        start_times = imu_data(1,1) : step : (imu_data(end,1) - win_len);

        % Pick windows near the start, middle, and end of activity
        mid_idx = round(length(start_times) / 2);
        preferred_idxs = unique([1:min(3, length(start_times)), ...
                         max(1, mid_idx-1):min(mid_idx+1, length(start_times)), ...
                         max(length(start_times)-2,1):length(start_times)]);

        
        preferred_idxs = preferred_idxs(preferred_idxs <= length(start_times));
        for i = 1:min(num_adl_windows, length(preferred_idxs))
            win_start = start_times(preferred_idxs(i));
            win_end = win_start + win_len;
            feature_vector = [];

            imu_sources = {'Back', 'Right_Thigh', 'Left_Thigh'};
            for s = 1:length(imu_sources)
                imu_name = imu_sources{s};
                if isfield(all_data.(participant).(trial), imu_name)
                    imu = all_data.(participant).(trial).(imu_name);
                    idx = imu(:,1) >= win_start & imu(:,1) <= win_end;
                    window = imu(idx,:);
                    for ch = 2:7
                        if size(window,2) >= ch
                            data = window(:,ch);
                            if ~isempty(data) && all(~isnan(data))
                                data = lowpass_filter_imu(data, Fs);
                                feature_vector = [feature_vector, feature_extract_591k(data, Fs)];
                            end
                        end
                    end
                end
            end
            % ECG
            if isfield(all_data.(participant).(trial), 'ECG')
                ecg = all_data.(participant).(trial).ECG;
                idx = ecg(:,1) >= win_start & ecg(:,1) <= win_end;
                window = ecg(idx,:);
                if size(window,2) >= 2
                    data = window(:,2);
                    if ~isempty(data) && all(~isnan(data))
                        data = bandpass_filter_ecg(data, Fs);
                        feature_vector = [feature_vector, feature_extract_591k(data, Fs)];
                    end
                end
            end
            % GSS
            if isfield(all_data.(participant).(trial), 'GSS')
                gss = all_data.(participant).(trial).GSS;
                idx = gss(:,1) >= win_start & gss(:,1) <= win_end;
                window = gss(idx,:);
                if size(window,2) >= 2
                    data = window(:,2);
                    if ~isempty(data) && all(~isnan(data))
                        data = bandpass_filter_ecg(data, Fs);
                        feature_vector = [feature_vector, feature_extract_591k(data, Fs)];
                    end
                end
            end

            if ~isempty(feature_vector) && abs(length(feature_vector) - expected_length) <= 10
                X = [X; feature_vector];
                y = [y; 0];  % or 0 for ADL
            end
        end
    end
end

%% Normalize features
X = zscore(X);

fprintf('\nTraining set summary:\n');
fprintf('  ADL (0):       %\n', sum(y == 0));
fprintf('  Near-Fall (1): %d\n', sum(y == 1));
fprintf('  Fall (2):      %d\n', sum(y == 2));
fprintf('Total:           %d windows\n', length(y));

% Train Random Forest (TreeBagger)
Mdl = TreeBagger(100, X, string(y), ...
    'Method', 'classification', ...
    'OOBPrediction', 'on', ...
    'ClassNames', ["0", "1", "2"], ...
    'Prior', 'empirical');  % Downweight ADL

% Save model
save('trained_GC_model.mat', 'Mdl');
%% predictions_cnn_FIXED.m
% Fixes:
% 1) Dynamic IMU selection (works for all test participants)
% 2) Filtering matches training
% 3) Exports only label 1/2 (NearFall/Fall) as required


load("GrandChallengeTestData.mat","test_data");
load("cnn_model.mat","net");

%% setup
Fs = 1000;
window_size_sec = 3;     % MUST match training
step_size_sec   = 1.0;   % can be 1 or 1.5 (denser gives more detections)

imu_sources_train = {'Back','Right_Thigh','Left_Thigh'};
imu_sources_test  = {'Sternum','Left_Arm','Right_Arm'};

output_dir = fullfile(pwd, 'CNN_pred_CSVs_FIXED');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

participants = fieldnames(test_data);

%% ===== SAFE FILTERS (same as training) =====
function y = safe_filtfilt(b,a,x)
    x = x(:);
    if numel(x) <= 24 || any(~isfinite(x)) || std(x) == 0
        y = x; return;
    end
    y = filtfilt(b,a,x);
end

function y = lpf_imu(x, Fs)
    [b,a] = butter(4, 10/(Fs/2), 'low');
    y = safe_filtfilt(b,a,x);
end

function y = bpf_ecg(x, Fs)
    [b,a] = butter(4, [0.5 40]/(Fs/2), 'bandpass');
    y = safe_filtfilt(b,a,x);
end

function y = lpf_gss(x, Fs)
    [b,a] = butter(4, 5/(Fs/2), 'low');
    y = safe_filtfilt(b,a,x);
end

%% Extraction (match train input) 
function seq = extract_sequence(data_struct, win_start, win_end, Fs, imu_sources)
    target_len = Fs * 3;
    seq = [];

    function sig_out = fix_length(sig)
        sig = sig(:);
        if length(sig) >= target_len
            sig_out = sig(1:target_len);
        else
            sig_out = [sig; zeros(target_len - length(sig),1)];
        end
    end

    % IMU: 3 sensors x 6 channels
    for s = 1:length(imu_sources)
        src = imu_sources{s};

        for ch = 2:7
            sig = [];
            if isfield(data_struct, src)
                imu = data_struct.(src);
                idx = imu(:,1)>=win_start & imu(:,1)<=win_end;
                window = imu(idx,:);
                if ~isempty(window) && size(window,2)>=ch
                    sig = window(:,ch);
                end
            end

            if isempty(sig) || numel(sig) < 25 || any(~isfinite(sig))
                sig = zeros(target_len,1);
            else
                sig = lpf_imu(sig, Fs);
            end

            sig = fix_length(sig);
            seq = [seq; sig']; %#ok<AGROW>
        end
    end

    % ECG
    sig = [];
    if isfield(data_struct,'ECG')
        ecg = data_struct.ECG;
        idx = ecg(:,1)>=win_start & ecg(:,1)<=win_end;
        if any(idx), sig = ecg(idx,2); end
    end

    if isempty(sig) || numel(sig) < 25 || any(~isfinite(sig))
        sig = zeros(target_len,1);
    else
        sig = bpf_ecg(sig, Fs);
    end
    sig = fix_length(sig);
    seq = [seq; sig'];

    % GSS
    sig = [];
    if isfield(data_struct,'GSS')
        gss = data_struct.GSS;
        idx = gss(:,1)>=win_start & gss(:,1)<=win_end;
        if any(idx), sig = gss(idx,2); end
    end

    if isempty(sig) || numel(sig) < 25 || any(~isfinite(sig))
        sig = zeros(target_len,1);
    else
        sig = lpf_gss(sig, Fs);
    end
    sig = fix_length(sig);
    seq = [seq; sig'];
end

%% Helper: Choose IMU set dynamically 
function [imu_sources, ref_sensor] = chooseImuSet(data_struct, imu_sources_test, imu_sources_train)
    if isfield(data_struct,'Sternum')
        imu_sources = imu_sources_test;
        ref_sensor = 'Sternum';
    elseif isfield(data_struct,'Back')
        imu_sources = imu_sources_train;
        ref_sensor = 'Back';
    else
        imu_sources = {};
        ref_sensor = "";
    end
end

%% Helper: Merge intervals  
function merged = mergeIntervals(output, max_duration)
    if isempty(output), merged = []; return; end
    output = sortrows(output,1);

    merged = [];
    current = output(1,:);

    for i = 2:size(output,1)
        nxt = output(i,:);

        % merge if overlapping/adjacent and same label
        if (nxt(1) - current(2) <= 0.01) && (nxt(3) == current(3))
            current(2) = max(current(2), nxt(2));
        else
            if (current(2)-current(1)) > max_duration
                current(2) = current(1) + max_duration;
            end
            merged = [merged; current]; %#ok<AGROW>
            current = nxt;
        end
    end

    if (current(2)-current(1)) > max_duration
        current(2) = current(1) + max_duration;
    end
    merged = [merged; current];
end

%% Run test 
for p = 1:length(participants)

    participant_id = participants{p};
    trials = fieldnames(test_data.(participant_id));

    for t = 1:length(trials)

        trial_name = trials{t};
        fprintf('Processing %s - %s\n', participant_id, trial_name);

        data_struct = test_data.(participant_id).(trial_name);

        [imu_sources, ref_sensor] = chooseImuSet(data_struct, imu_sources_test, imu_sources_train);
        if isempty(imu_sources) || ref_sensor == ""
            fprintf("  Skipped (no recognized IMU fields)\n");
            continue;
        end

        imu_data = data_struct.(ref_sensor);
        if isempty(imu_data) || size(imu_data,1) < Fs*3
            fprintf("  Skipped (IMU too short)\n");
            continue;
        end

        tvec = imu_data(:,1);
        if isempty(tvec) || numel(tvec) < 2
            continue;
        end

        start_times = tvec(1):step_size_sec:(tvec(end)-window_size_sec);
        if isempty(start_times)
            continue;
        end

        Xtest = {};
        time_ranges = [];

        % build window sequences
        for w = 1:length(start_times)
            win_start = start_times(w);
            win_end   = win_start + window_size_sec;

            seq = extract_sequence(data_struct, win_start, win_end, Fs, imu_sources);
            Xtest{end+1,1} = seq; %#ok<AGROW>
            time_ranges(end+1,:) = [win_start, win_end]; %#ok<AGROW>
        end

        if isempty(Xtest)
            continue;
        end

        % cnn prediction
        YPred  = classify(net, Xtest);
        scores = predict(net, Xtest);

        % Map categorical to numeric: ADL=0, NearFall=1, Fall=2
        % Order matches training categorical labels
        predicted_labels = double(YPred) - 1;

        % Confidence gating (optional): low confidence -> ADL
        conf_max = max(scores,[],2);
        predicted_labels(conf_max < 0.80) = 0;

        % Keep only NearFall/Fall for output (as required)
        keep_idx = predicted_labels ~= 0;
        output = [time_ranges(keep_idx,:), predicted_labels(keep_idx)];

        if isempty(output)
            fprintf("  No events detected.\n");
            continue;
        end

        % Merge windows into intervals
        max_duration = 4; % cap event duration
        merged = mergeIntervals(output, max_duration);

        % Save CSV with required naming P####_T##.csv
        tok = regexp(trial_name, 'T\d\d', 'match', 'once');
        if isempty(tok)
            % fallback: extract digits
            trial_id = regexp(trial_name,'\d+','match','once');
            tok = sprintf("T%02d", str2double(trial_id));
        end

        filename = sprintf('%s_%s.csv', participant_id, tok);
        writematrix(merged, fullfile(output_dir, filename));

        fprintf('  %d events saved -> %s\n', size(merged,1), filename);
    end
end

disp("Done. CSVs saved in CNN_pred_CSVs_FIXED");

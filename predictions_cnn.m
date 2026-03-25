%% load data
load("GrandChallengeTestData.mat")
load("cnn_model.mat")
%% setup
Fs = 1000;
window_size_sec = 3;     % MUST match training
step_size_sec = 1;

imu_sources_default = {'Back','Right_Thigh','Left_Thigh'};
imu_sources_alt = {'Sternum','Left_Arm','Right_Arm'};

output_dir = fullfile(pwd, 'CNN_pred_CSVs');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

participants = fieldnames(test_data);

%% extraction to match the train input
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

    % IMU
    for s = 1:length(imu_sources)
        src = imu_sources{s};

        for ch = 2:7
            if isfield(data_struct, src)
                imu = data_struct.(src);
                idx = imu(:,1)>=win_start & imu(:,1)<=win_end;
                window = imu(idx,:);

                if size(window,2)>=ch
                    sig = window(:,ch);
                else
                    sig = [];
                end
            else
                sig = [];
            end

            if length(sig) < 25 || any(isnan(sig))
                sig = zeros(target_len,1);
            end

            sig = fix_length(sig);
            seq = [seq; sig'];
        end
    end

    % ECG
    if isfield(data_struct,'ECG')
        ecg = data_struct.ECG;
        idx = ecg(:,1)>=win_start & ecg(:,1)<=win_end;
        sig = ecg(idx,2);
    else
        sig = [];
    end

    if length(sig) < 25
        sig = zeros(target_len,1);
    end

    sig = fix_length(sig);
    seq = [seq; sig'];

    % GSS
    if isfield(data_struct,'GSS')
        gss = data_struct.GSS;
        idx = gss(:,1)>=win_start & gss(:,1)<=win_end;
        sig = gss(idx,2);
    else
        sig = [];
    end

    if length(sig) < 25
        sig = zeros(target_len,1);
    end

    sig = fix_length(sig);
    seq = [seq; sig'];
end

%% run test
for p = 1:length(participants)

    participant_id = participants{p};
    trials = fieldnames(test_data.(participant_id));

    for t = 1:length(trials)

        trial_name = trials{t};
        fprintf('Processing %s - %s\n', participant_id, trial_name);

        data_struct = test_data.(participant_id).(trial_name);

        % Select IMU set
        if strcmp(participant_id, 'P1544')
            imu_sources = imu_sources_alt;
            ref_sensor = 'Sternum';
        else
            imu_sources = imu_sources_default;
            ref_sensor = 'Back';
        end

        if ~isfield(data_struct, ref_sensor)
            continue;
        end

        imu_data = data_struct.(ref_sensor);

        if isempty(imu_data) || size(imu_data,1) < Fs*3
            continue;
        end

        tvec = imu_data(:,1);

        start_times = tvec(1):step_size_sec:(tvec(end)-window_size_sec);

        if isempty(start_times)
            continue;
        end

        Xtest = {};
        time_ranges = [];

        %% build
        for w = 1:length(start_times)

            win_start = start_times(w);
            win_end = win_start + window_size_sec;

            seq = extract_sequence(data_struct, win_start, win_end, Fs, imu_sources);

            if isempty(seq)
                continue;
            end

            Xtest{end+1,1} = seq;
            time_ranges(end+1,:) = [win_start, win_end];
        end

        if isempty(Xtest)
            continue;
        end

        %% cnn prediction
        YPred = classify(net, Xtest);
        scores = predict(net, Xtest);

        predicted_labels = double(YPred) - 1; % ADL=0, NearFall=1, Fall=2

        %% confidence
        conf_max = max(scores,[],2);
        predicted_labels(conf_max < 0.8) = 0;

        %% remove adl
        keep_idx = predicted_labels ~= 0;

        output = [time_ranges(keep_idx,:), predicted_labels(keep_idx)];

        if isempty(output)
            continue;
        end

        %% sort and merge
        output = sortrows(output,1);
        merged = [];
        current = output(1,:);
        max_duration = 4;

        for i = 2:size(output,1)
            next = output(i,:);

            if next(1) - current(2) <= 0.01 && next(3)==current(3)
                current(2) = max(current(2), next(2));
            else
                if current(2)-current(1) > max_duration
                    current(2) = current(1)+max_duration;
                end
                merged = [merged; current];
                current = next;
            end
        end

        merged = [merged; current];

        %% save csvs
        trial_id = regexp(trial_name,'\d+','match','once');
        filename = sprintf('%s_T%02s.csv', participant_id, trial_id);

        writematrix(merged, fullfile(output_dir, filename));

        fprintf('%s - %s | %d events saved\n', participant_id, trial_name, size(merged,1));
    end
end

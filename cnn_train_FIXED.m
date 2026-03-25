%% cnn_train_FIXED.m
% Fixes:
% 1) Adds ADL windows to training set (was missing before)
% 2) Robust IMU timebase (does not require Back)
% 3) Consistent safe filtering
% 4) Keeps same model structure + window size (3 s)

%% load
load("GrandChallengeData.mat");  % all_data, clean_labels

%% setup
Fs = 1000;
window_size_sec = 3;
target_len = Fs * window_size_sec;

imu_sources = {'Back','Right_Thigh','Left_Thigh'};   % training IMUs
participants = fieldnames(clean_labels);

X = {};     % each cell: [numChannels x target_len]
Y = [];     % numeric labels: 0 ADL, 1 NearFall, 2 Fall

%% SAFE FILTERS
function y = safe_filtfilt(b,a,x)
    x = x(:);
    if numel(x) <= 24 || any(~isfinite(x)) || std(x) == 0
        y = x;  % too short or flat -> no filtering
        return;
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

%% EXTRACTION (match CNN input)
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

    % IMU: (3 sensors * 6 channels = 18)
    for s = 1:length(imu_sources)
        src = imu_sources{s};

        for ch = 2:7
            sig = [];

            if isfield(data_struct, src)
                imu = data_struct.(src);
                idx = imu(:,1) >= win_start & imu(:,1) <= win_end;
                window = imu(idx,:);

                if ~isempty(window) && size(window,2) >= ch
                    sig = window(:,ch);
                end
            end

            if isempty(sig) || numel(sig) < 25 || any(isnan(sig)) || any(~isfinite(sig))
                sig = zeros(target_len,1);
            else
                sig = lpf_imu(sig, Fs);
                sig = fix_length(sig);
            end

            sig = fix_length(sig);
            seq = [seq; sig']; %#ok<AGROW>
        end
    end

    % ECG: 1 channel
    sig = [];
    if isfield(data_struct,'ECG')
        ecg = data_struct.ECG;
        idx = ecg(:,1) >= win_start & ecg(:,1) <= win_end;
        if any(idx), sig = ecg(idx,2); end
    end
    if isempty(sig) || numel(sig) < 25 || any(~isfinite(sig))
        sig = zeros(target_len,1);
    else
        sig = bpf_ecg(sig, Fs);
    end
    sig = fix_length(sig);
    seq = [seq; sig'];

    % GSS: 1 channel
    sig = [];
    if isfield(data_struct,'GSS')
        gss = data_struct.GSS;
        idx = gss(:,1) >= win_start & gss(:,1) <= win_end;
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

%% BUILD EVENT WINDOWS (NearFall/Fall)
fprintf("Extracting Event Windows... ");
total = length(participants);

for p = 1:length(participants)
    participant = participants{p};
    if ~isfield(all_data, participant), continue; end

    trials = fieldnames(clean_labels.(participant));

    for t = 1:length(trials)
        trial = trials{t};

        data_trials = fieldnames(all_data.(participant));
        match_idx = find(contains(data_trials, trial,'IgnoreCase',true), 1);
        if isempty(match_idx), continue; end
        data_trial = data_trials{match_idx};

        labels = clean_labels.(participant).(trial);
        if isempty(labels), continue; end

        % keep same grouping as teammate (works with their pipeline)
        group_id = [1; cumsum(diff(labels(:,3))~=0)+1];
        groups = unique(group_id);

        for g = groups'
            idx = group_id==g;
            grp_labels = labels(idx,1);
            grp_times  = labels(idx,2);

            if any(grp_labels==4)
                event_time = mean(grp_times(grp_labels==4))/1000;
                label = 2; % Fall
            elseif any(grp_labels==2)
                event_time = mean(grp_times(grp_labels==2))/1000;
                label = 1; % Near-Fall
            else
                continue;
            end

            win_start = max(0, event_time - window_size_sec/2);
            win_end   = win_start + window_size_sec;

            seq = extract_sequence(all_data.(participant).(data_trial), ...
                                   win_start, win_end, Fs, imu_sources);

            X{end+1,1} = seq; %#ok<AGROW>
            Y(end+1,1) = label; %#ok<AGROW>
        end
    end

    prog = 100*(p/total);
    fprintf('\b\b\b\b%3.0f%%', prog);
end
fprintf("\n");

%% ADD ADL WINDOWS (FIXED)
fprintf("Extracting ADL Windows... ");

adl_keywords = {'Stand','Walk','Sit','Lie','Stairs','Jog','Pick','JJ'}; % include JJ if present
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

        % Use any available IMU in this struct as timebase (no Back requirement)
        imuAvail = imu_sources(ismember(imu_sources, fieldnames(data_struct)));
        if isempty(imuAvail), continue; end

        imu0 = data_struct.(imuAvail{1});
        if isempty(imu0) || size(imu0,1) < 2, continue; end

        tvec = imu0(:,1);
        if isempty(tvec) || numel(tvec) < 2, continue; end
        if (tvec(end) - tvec(1)) < window_size_sec, continue; end

        start_times = tvec(1):adl_stride:(tvec(end)-window_size_sec);
        if isempty(start_times), continue; end

        nStarts = length(start_times);
        mid = round(nStarts/2);
        idxs = unique([1:min(3,nStarts), mid-1:mid+1, max(nStarts-2,1):nStarts]);
        idxs = idxs(idxs>=1 & idxs<=nStarts);

        for i = 1:min(num_adl_windows, length(idxs))
            win_start = start_times(idxs(i));
            win_end   = win_start + window_size_sec;

            seq = extract_sequence(data_struct, win_start, win_end, Fs, imu_sources);
            X{end+1,1} = seq; %#ok<AGROW>
            Y(end+1,1) = 0;   %#ok<AGROW>  % ADL
        end
    end
end
fprintf("done.\n");

%% labels
Y = categorical(Y, [0 1 2], ["ADL","NearFall","Fall"]);

%% train and val split (random windows)
rng(1);
idx = randperm(length(Y));
split = round(0.8*length(Y));

XTrain = X(idx(1:split));
YTrain = Y(idx(1:split));

XVal = X(idx(split+1:end));
YVal = Y(idx(split+1:end));

%% CNN
numChannels = size(XTrain{1},1);

layers = [
    sequenceInputLayer(numChannels)

    convolution1dLayer(5,32,'Padding','same')
    reluLayer
    layerNormalizationLayer

    convolution1dLayer(5,32,'Padding','same')
    reluLayer
    layerNormalizationLayer

    globalAveragePooling1dLayer

    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer
];

options = trainingOptions("adam", ...
    "MiniBatchSize", 10, ...
    "MaxEpochs", 30, ...
    "SequencePaddingDirection","left", ...
    "ValidationData",{XVal,YVal}, ...
    "ValidationFrequency", 10, ...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "Verbose",1);

net = trainNetwork(XTrain, YTrain, layers, options);
save("cnn_model.mat","net");

%% metrics
YPred = classify(net, XVal);
acc = mean(YPred == YVal);

confMat = confusionmat(YVal, YPred);

TP = diag(confMat);
FP = sum(confMat,1)' - TP;
FN = sum(confMat,2) - TP;
TN = sum(confMat(:)) - (TP+FP+FN);

sensitivity = TP ./ (TP + FN);
specificity = TN ./ (TN + FP);

fprintf("\n=== CNN RESULTS (FIXED) ===\n");
fprintf("Accuracy: %.2f%%\n", acc*100);

classNames = categories(YVal);
for i = 1:length(TP)
    fprintf("%s -> Sensitivity: %.2f%% | Specificity: %.2f%%\n", ...
        string(classNames{i}), sensitivity(i)*100, specificity(i)*100);
end

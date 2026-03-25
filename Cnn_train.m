%% load
load("GrandChallengeData.mat");

%% setup
Fs = 1000;
window_size_sec = 3;
target_len = Fs * window_size_sec;

imu_sources = {'Back','Right_Thigh','Left_Thigh'};
participants = fieldnames(clean_labels);

X = {};
Y = [];

%% filter
function y = lpf_imu(x, Fs)
    if length(x) < 25, y = zeros(3000,1); return; end
    [b,a] = butter(4,10/(Fs/2),'low');
    y = filtfilt(b,a,x);
end

function y = bpf_ecg(x, Fs)
    if length(x) < 25, y = zeros(3000,1); return; end
    [b,a] = butter(4,[0.5 40]/(Fs/2),'bandpass');
    y = filtfilt(b,a,x);
end

function y = lpf_gss(x, Fs)
    if length(x) < 25, y = zeros(3000,1); return; end
    [b,a] = butter(4,5/(Fs/2),'low');
    y = filtfilt(b,a,x);
end

%% extraction
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

    % IMU (3 sensors × 6 channels = 18)
    for s = 1:length(imu_sources)
        src = imu_sources{s};

        for ch = 2:7
            if isfield(data_struct, src)
                imu = data_struct.(src);
                idx = imu(:,1)>=win_start & imu(:,1)<=win_end;
                window = imu(idx,:);

                if size(window,2)>=ch
                    sig = window(:,ch);
                    if length(sig)>=25 && all(~isnan(sig))
                        sig = lpf_imu(sig, Fs);
                    else
                        sig = zeros(target_len,1);
                    end
                else
                    sig = zeros(target_len,1);
                end
            else
                sig = zeros(target_len,1);
            end

            sig = fix_length(sig);
            seq = [seq; sig'];
        end
    end

    % ECG (1 channel)
    if isfield(data_struct,'ECG')
        ecg = data_struct.ECG;
        idx = ecg(:,1)>=win_start & ecg(:,1)<=win_end;
        sig = ecg(idx,2);
        if length(sig)>=25
            sig = bpf_ecg(sig, Fs);
        else
            sig = zeros(target_len,1);
        end
    else
        sig = zeros(target_len,1);
    end
    sig = fix_length(sig);
    seq = [seq; sig'];

    % GSS (1 channel)
    if isfield(data_struct,'GSS')
        gss = data_struct.GSS;
        idx = gss(:,1)>=win_start & gss(:,1)<=win_end;
        sig = gss(idx,2);
        if length(sig)>=25
            sig = lpf_gss(sig, Fs);
        else
            sig = zeros(target_len,1);
        end
    else
        sig = zeros(target_len,1);
    end
    sig = fix_length(sig);
    seq = [seq; sig'];
end

%% dataset
fprintf("Extracting Training Set...    ")

total = length(participants);

for p = 1:length(participants)
    participant = participants{p};
    if ~isfield(all_data, participant), continue; end

    trials = fieldnames(clean_labels.(participant));

    for t = 1:length(trials)
        trial = trials{t};

        data_trials = fieldnames(all_data.(participant));
        match_idx = find(contains(data_trials, trial,'IgnoreCase',true),1);
        if isempty(match_idx), continue; end
        data_trial = data_trials{match_idx};

        labels = clean_labels.(participant).(trial);
        if isempty(labels), continue; end

        group_id = [1; cumsum(diff(labels(:,3))~=0)+1];
        groups = unique(group_id);

        for g = groups'
            idx = group_id==g;
            grp_labels = labels(idx,1);
            grp_times = labels(idx,2);

            if any(grp_labels==4)
                event_time = mean(grp_times(grp_labels==4))/1000;
                label = 2; % Fall
            elseif any(grp_labels==2)
                event_time = mean(grp_times(grp_labels==2))/1000;
                label = 1; % Near-Fall
            else
                continue;
            end

            win_start = max(0,event_time-window_size_sec/2);
            win_end = win_start + window_size_sec;

            seq = extract_sequence( ...
                all_data.(participant).(data_trial), ...
                win_start, win_end, Fs, imu_sources);

            X{end+1,1} = seq;
            Y(end+1,1) = label;
        end
    end

    prog = 100*(p/total);
    fprintf('\b\b\b\b%3.0f%%',prog);
end
fprintf("\n")

%% adl window
adl_keywords = {'Stand','Walk','Sit','Lie','Stairs','Jog'};

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

            % Safety check
            if isempty(tvec) || length(tvec) < 2
                continue;
            end
            
            if (tvec(end) - tvec(1)) < window_size_sec
                continue;
            end
            
            start_times = tvec(1):1.5:(tvec(end)-window_size_sec);
            
            if isempty(start_times)
                continue;
            end
    end
end
%% labels
Y = categorical(Y, [0 1 2], ["ADL","NearFall","Fall"]);

%% train and val
rng(1)
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

%% TRAINING OPTIONS from assignment 6
options = trainingOptions("adam", ...
    "MiniBatchSize", 10, ...
    "MaxEpochs", 30, ...
    "SequencePaddingDirection","left", ...
    "ValidationData",{XVal,YVal}, ...
    "ValidationFrequency", 10, ...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "Verbose",1);

net = trainNetwork(XTrain,YTrain,layers,options);

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

fprintf("\n=== CNN RESULTS ===\n");
fprintf("Accuracy: %.2f%%\n", acc*100);

classNames = categories(YVal);

for i = 1:length(TP)
    fprintf("%s -> Sensitivity: %.2f%% | Specificity: %.2f%%\n", ...
        string(classNames{i}), sensitivity(i)*100, specificity(i)*100);
end

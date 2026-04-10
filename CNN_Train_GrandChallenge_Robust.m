% Location-robust CNN for ADL / NearFall / Fall using IMU magnitude aggregation (+ optional ECG/GSS)
% Saves: cnn_model_gc.mat
clear; clc; close all;

%% Load training data 
S = load("GrandChallengeData.mat","all_data","clean_labels");
all_data = S.all_data;
clean_labels = S.clean_labels;

Fs = 1000;
winLen_s = 3.0;
winLen_N = round(winLen_s * Fs);

% ADL sampling 
adl_stride_s = 1.5;

% Use this union to automatically find available IMUs in each trial
imu_union = {'Back','Left_Thigh','Right_Thigh','Sternum','Left_Arm','Right_Arm'};

% Toggle these if you want IMU-only vs IMU+ECG+GSS
USE_ECG = true;
USE_GSS = true;

% Output labels: 0 ADL, 1 NearFall, 2 Fall
X = {};
Y = [];

%% Filters
[bIMU,aIMU] = butter(4, 10/(Fs/2), 'low');          % IMU low-pass
[bECG,aECG] = butter(4, [0.5 40]/(Fs/2), 'bandpass');% ECG band-pass
[bGSS,aGSS] = butter(4, 5/(Fs/2), 'low');          % GSS low-pass

safeFilt = @(b,a,x) (numel(x) > 24 && all(isfinite(x)) && std(x)>0) ...
    * filtfilt(b,a,x) + ...
    ~(numel(x) > 24 && all(isfinite(x)) && std(x)>0) ...
    * x;

%% Helper: Build robust multi-channel sequence 
% Output: seq is [C x T] where T = 3000 (3 s @ 1000 Hz)
function seq = build_seq(data_struct, t1, t2, Fs, imu_union, winLen_N, safeFilt, bIMU,aIMU,bECG,aECG,bGSS,aGSS, USE_ECG, USE_GSS)

    fields = fieldnames(data_struct);
    imuAvail = imu_union(ismember(imu_union, fields));

    accMags = [];
    gyrMags = [];

    % IMU: For each available IMU, compute acc_mag and gyro_mag
    for s = 1:numel(imuAvail)
        imu = data_struct.(imuAvail{s});
        if isempty(imu) || size(imu,2) < 7, continue; end

        idx = imu(:,1) >= t1 & imu(:,1) <= t2;
        w = imu(idx,:);
        if size(w,1) < 50, continue; end

        acc = w(:,2:4);  % ax ay az
        gyr = w(:,5:7);  % gx gy gz

        % filter per axis then magnitude
        accF = zeros(size(acc));
        gyrF = zeros(size(gyr));
        for k=1:3
            accF(:,k) = safeFilt(bIMU,aIMU, acc(:,k));
            gyrF(:,k) = safeFilt(bIMU,aIMU, gyr(:,k));
        end

        acc_mag = sqrt(sum(accF.^2,2));
        gyr_mag = sqrt(sum(gyrF.^2,2));

        accMags = [accMags, fixLen(acc_mag, winLen_N)]; %#ok<AGROW>
        gyrMags = [gyrMags, fixLen(gyr_mag, winLen_N)]; %#ok<AGROW>
    end

    % If no IMU present, return empty (we require IMU)
    if isempty(accMags) && isempty(gyrMags)
        seq = [];
        return;
    end

    % Aggregate across IMUs: mean and max (robust to placement differences)
    acc_mean = mean(accMags, 2);
    acc_max  = max(accMags, [], 2);

    gyr_mean = mean(gyrMags, 2);
    gyr_max  = max(gyrMags, [], 2);

    chans = [acc_mean, acc_max, gyr_mean, gyr_max]; % 4 channels

    % ECG 
    if USE_ECG
        if isfield(data_struct,'ECG') && ~isempty(data_struct.ECG)
            ecg = data_struct.ECG;
            idx = ecg(:,1) >= t1 & ecg(:,1) <= t2;
            sig = ecg(idx,2);
            sig = fixLen(sig(:), winLen_N);
            sig = safeFilt(bECG,aECG,sig);
        else
            sig = zeros(numel(acc_mean),1);
        end
        chans = [chans, sig]; %#ok<AGROW>
    end

    % GSS 
    if USE_GSS
        if isfield(data_struct,'GSS') && ~isempty(data_struct.GSS)
            gss = data_struct.GSS;
            idx = gss(:,1) >= t1 & gss(:,1) <= t2;
            sig = gss(idx,2);
            sig = fixLen(sig(:), winLen_N);
            sig = safeFilt(bGSS,aGSS,sig);
        else
            sig = zeros(numel(acc_mean),1);
        end
        chans = [chans, sig]; %#ok<AGROW>
    end

    % Return as [numChannels x winLen_N]
    seq = chans.'; 
    mu = mean(seq, 2);
    sd = std(seq, 0, 2);
    sd(sd < 1e-6) = 1;
    seq = (seq - mu) ./ sd;
end

%% Build EVENT windows from clean_labels 
participants = fieldnames(clean_labels);
fprintf("Extracting EVENT windows... ");

for p = 1:numel(participants)
    pid = participants{p};
    if ~isfield(all_data,pid), continue; end

    acts = fieldnames(clean_labels.(pid));

    for a = 1:numel(acts)
        act = acts{a};
        timing = clean_labels.(pid).(act);
        if isempty(timing), continue; end

        % Find matching trial name in all_data
        data_trials = fieldnames(all_data.(pid));
        match_idx = find(contains(data_trials, act, 'IgnoreCase', true), 1);
        if isempty(match_idx), continue; end
        trialName = data_trials{match_idx};

        data_struct = all_data.(pid).(trialName);

        evt = timing(:,1);
        tms = timing(:,2) / 1000; % seconds

        % group by event id col3 if present, else just proceed
        if size(timing,2) >= 3
            group_id = [1; cumsum(diff(timing(:,3))~=0)+1];
            groups = unique(group_id);
        else
            group_id = ones(size(evt));
            groups = 1;
        end

        for g = groups'
            idxg = group_id==g;
            grp_labels = evt(idxg);
            grp_times  = tms(idxg);

            if any(grp_labels==4)
                event_time = mean(grp_times(grp_labels==4));
                lab = 2; % Fall
            elseif any(grp_labels==2)
                event_time = mean(grp_times(grp_labels==2));
                lab = 1; % NearFall
            else
                continue;
            end

            t1 = max(0, event_time - winLen_s/2);
            t2 = t1 + winLen_s;

            seq = build_seq(data_struct, t1, t2, Fs, imu_union, winLen_N, safeFilt, ...
                            bIMU,aIMU,bECG,aECG,bGSS,aGSS, USE_ECG, USE_GSS);

            if isempty(seq), continue; end

            X{end+1,1} = seq; %#ok<AGROW>
            Y(end+1,1) = lab; %#ok<AGROW>
        end
    end

    if mod(p,3)==0, fprintf("."); end
end
fprintf(" done.\n");

%% Build ADL windows from ADL trials in all_data 
adl_keywords = {'Stand','Walk','Sit','Lie','Stairs','Jog','Pick','JJ'};
fprintf("Extracting ADL windows... ");

for p = 1:numel(participants)
    pid = participants{p};
    if ~isfield(all_data,pid), continue; end

    trials = fieldnames(all_data.(pid));

    for t = 1:numel(trials)
        trialName = trials{t};
        if ~any(contains(trialName, adl_keywords)), continue; end

        data_struct = all_data.(pid).(trialName);

        % pick any available IMU as timebase
        fields = fieldnames(data_struct);
        imuAvail = imu_union(ismember(imu_union, fields));
        if isempty(imuAvail), continue; end
        imu0 = data_struct.(imuAvail{1});
        if isempty(imu0) || size(imu0,1) < 2, continue; end
        tvec = imu0(:,1);

        if (tvec(end)-tvec(1)) < winLen_s, continue; end
        start_times = tvec(1):adl_stride_s:(tvec(end)-winLen_s);
        if isempty(start_times), continue; end

        % sample a few windows per trial (keeps dataset size reasonable)
        idxs = unique(round(linspace(1, numel(start_times), min(5,numel(start_times)))));
        for ii = 1:numel(idxs)
            t1 = start_times(idxs(ii));
            t2 = t1 + winLen_s;

            seq = build_seq(data_struct, t1, t2, Fs, imu_union, winLen_N, safeFilt, ...
                            bIMU,aIMU,bECG,aECG,bGSS,aGSS, USE_ECG, USE_GSS);
            if isempty(seq), continue; end

            X{end+1,1} = seq; %#ok<AGROW>
            Y(end+1,1) = 0;   %#ok<AGROW>
        end
    end
end
fprintf(" done.\n");

%% Labels 
Y = categorical(Y, [0 1 2], ["ADL","NearFall","Fall"]);

%% Train/Val split 
rng(1);
idx = randperm(numel(Y));
split = round(0.8*numel(Y));

XTrain = X(idx(1:split));
YTrain = Y(idx(1:split));
XVal   = X(idx(split+1:end));
YVal   = Y(idx(split+1:end));

numChannels = size(XTrain{1},1);

%% CNN (adapted from Assignment6 ECG template structure) 
layers = [
  sequenceInputLayer(numChannels)

  convolution1dLayer(7,64,'Padding','same')
  batchNormalizationLayer
  reluLayer
  dropoutLayer(0.2)

  convolution1dLayer(5,64,'Padding','same')
  batchNormalizationLayer
  reluLayer
  dropoutLayer(0.2)

  globalAveragePooling1dLayer
  fullyConnectedLayer(3)
  softmaxLayer
  classificationLayer
];

options = trainingOptions("adam", ...
    "MiniBatchSize", 10, ...
    "MaxEpochs", 30, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", {XVal, YVal}, ...
    "ValidationFrequency", 10, ...
    "InitialLearnRate", 1e-3, ...
    "LearnRateSchedule", "piecewise", ...
    "LearnRateDropFactor", 0.5, ...
    "LearnRateDropPeriod", 10, ...
    "Plots", "training-progress", ...
    "Verbose", true);

net = trainNetwork(XTrain, YTrain, layers, options);

%% Validation metrics 
YPred = classify(net, XVal);
acc = mean(YPred == YVal);
fprintf("\nCNN Validation Accuracy: %.4f\n", acc);

confMat = confusionmat(YVal, YPred);
disp("Confusion matrix:"); disp(confMat);

save("cnn_model_gc.mat","net","USE_ECG","USE_GSS","winLen_s","Fs","imu_union");
%% Local function: Fixed-length padding/truncation 
function sig_out = fixLen(sig, winLen_N)
    sig = sig(:);
    if numel(sig) >= winLen_N
        sig_out = sig(1:winLen_N);
    else
        sig_out = [sig; zeros(winLen_N-numel(sig),1)];
    end
end
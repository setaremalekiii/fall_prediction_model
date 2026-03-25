% Generates required prediction CSVs from GrandChallengeTestData.mat using trained TreeBagger model
% Output format per file: [start_s, end_s, label] with label 1=near-fall, 2=fall (no header)

%% Settings 
testMatPath   = "GrandChallengeTestData.mat";
modelPath     = "trained_model.mat";
outDir        = "TB_pred_CSVs";
if ~exist(outDir,"dir"), mkdir(outDir); end

Fs = 1000;                % consistent with  pipeline
winLen_s  = 3.0;           % MUST match training
step_s    = 1.5;           % match ADL stride used in training
minN      = 50;            % minimum samples for feature extraction in a window
confThresh = 0.75;         % optional confidence gate (lower = more events)

imu_sources_union = {'Sternum','Left_Arm','Right_Arm','Back','Left_Thigh','Right_Thigh'};

%% Load model + test data 
M = load(modelPath);
Mdl = M.Mdl;

T = load(testMatPath,"test_data");
test_data = T.test_data;

participants = fieldnames(test_data);
fprintf("Loaded %d test participants.\n", numel(participants));

%% Safe filtering 
function y = safe_filtfilt(b,a,x)
    x = x(:);
    if numel(x) <= 24 || any(~isfinite(x)) || std(x)==0
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

%% Safe feature wrapper (prevents FFT edge cases) 
function fv = safe_feature_extract(sig, Fs, minN)
    if isempty(sig) || numel(sig) < minN || any(~isfinite(sig)) || std(sig)==0
        fv = [];
        return;
    end
    fv = feature_extract_591k(sig, Fs);
end

%% Robust window feature builder (same logic as training) 
function fv = extract_features_from_window(data_struct, win_start, win_end, Fs, imu_sources_union, minN)

    fv = [];

    % IMU magnitude features per sensor 
    accFeatAll = [];
    gyrFeatAll = [];

    % find IMU fields that actually exist for this trial
    fields = fieldnames(data_struct);
    imuAvail = imu_sources_union(ismember(imu_sources_union, fields));

    for s = 1:numel(imuAvail)
        src = imuAvail{s};
        imu = data_struct.(src);

        idx = imu(:,1) >= win_start & imu(:,1) <= win_end;
        window = imu(idx,:);

        if isempty(window) || size(window,2) < 7
            continue;
        end

        acc = window(:,2:4);
        gyr = window(:,5:7);

        accF = lpf_imu(acc, Fs);
        gyrF = lpf_imu(gyr, Fs);

        acc_mag = sqrt(sum(accF.^2,2));
        gyr_mag = sqrt(sum(gyrF.^2,2));

        f_acc = safe_feature_extract(acc_mag, Fs, minN);
        f_gyr = safe_feature_extract(gyr_mag, Fs, minN);

        if ~isempty(f_acc), accFeatAll(end+1,:) = f_acc; end %#ok<AGROW>
        if ~isempty(f_gyr), gyrFeatAll(end+1,:) = f_gyr; end %#ok<AGROW>
    end

    % Require IMU features (primary modality)
    if isempty(accFeatAll) && isempty(gyrFeatAll)
        fv = [];
        return;
    end

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

        if numel(sig) >= minN && all(isfinite(sig)) && std(sig)>0
            sig = bpf_ecg(sig, Fs);
            f_ecg = safe_feature_extract(sig, Fs, minN);
            if ~isempty(f_ecg), fv = [fv, f_ecg]; end
        end
    end

    % GSS 
    if isfield(data_struct,'GSS')
        gss = data_struct.GSS;
        idx = gss(:,1) >= win_start & gss(:,1) <= win_end;
        sig = gss(idx,2);

        if numel(sig) >= minN && all(isfinite(sig)) && std(sig)>0
            sig = lpf_gss(sig, Fs);
            f_gss = safe_feature_extract(sig, Fs, minN);
            if ~isempty(f_gss), fv = [fv, f_gss]; end
        end
    end
end

%% Helper Merge intervals
function merged = mergeIntervals(events, gapTol)
    % events: [start end label]
    if isempty(events)
        merged = [];
        return;
    end

    events = sortrows(events,1);
    merged = events(1,:);

    for i = 2:size(events,1)
        cur = merged(end,:);
        nxt = events(i,:);

        % overlap/adjacent?
        if nxt(1) <= cur(2) + gapTol
            % merge interval, severity wins (2 beats 1)
            merged(end,2) = max(cur(2), nxt(2));
            merged(end,3) = max(cur(3), nxt(3));
        else
            merged(end+1,:) = nxt; %#ok<AGROW>
        end
    end
end

%% Helper Write CSV  
function writeCSV(outDir, pid, trialName, out)
    tok = regexp(trialName, 'T\d\d[A-Z]?', 'match', 'once');  % matches T01, T01A, T08B
    if isempty(tok)
        tok = regexp(trialName, 'T\d\d', 'match', 'once');
    end
    fname = fullfile(outDir, sprintf("%s_%s.csv", pid, tok));

    % required: no header
    if isempty(out)
        writematrix([], fname);
    else
        writematrix(out, fname);
    end
end

%% Main loop 
for p = 1:numel(participants)
    pid = participants{p};
    trials = fieldnames(test_data.(pid));

    for t = 1:numel(trials)
        trialName = trials{t};
        data_struct = test_data.(pid).(trialName);

        % Choose time base from any available IMU field
        fields = fieldnames(data_struct);
        imuAvail = imu_sources_union(ismember(imu_sources_union, fields));
        if isempty(imuAvail)
            writeCSV(outDir, pid, trialName, []);
            continue;
        end

        imu0 = data_struct.(imuAvail{1});
        if isempty(imu0) || size(imu0,1) < 2
            writeCSV(outDir, pid, trialName, []);
            continue;
        end

        tvec = imu0(:,1);
        if (tvec(end) - tvec(1)) < winLen_s
            writeCSV(outDir, pid, trialName, []);
            continue;
        end

        start_times = tvec(1):step_s:(tvec(end) - winLen_s);

        Xtest = [];
        time_ranges = [];

        for w = 1:numel(start_times)
            win_start = start_times(w);
            win_end   = win_start + winLen_s;

            fv = extract_features_from_window(data_struct, win_start, win_end, Fs, imu_sources_union, minN);
            if isempty(fv)
                continue;
            end

            Xtest = [Xtest; fv]; %#ok<AGROW>
            time_ranges = [time_ranges; win_start win_end]; %#ok<AGROW>
        end

        if isempty(Xtest)
            writeCSV(outDir, pid, trialName, []);
            continue;
        end

        % Normalize: since you used zscore in training but didn't save mu/sigma,
        % we standardize per-trial feature matrix as a practical fallback.
        Xtest = zscore(Xtest);

        % Predict labels (0/1/2)
        [predStr, scores] = predict(Mdl, Xtest);
        pred = str2double(predStr);

        % Optional confidence gate: low confidence -> ADL (0)
        confMax = max(scores,[],2);
        pred(confMax < confThresh) = 0;

        % Require >=2 consecutive windows for label 1/2 (removes one-off spikes)
        pred2 = pred;
        for k = 1:numel(pred)
            if pred(k)==0, continue; end
            left  = (k>1) && (pred(k-1)==pred(k));
            right = (k<numel(pred)) && (pred(k+1)==pred(k));
            if ~(left || right)
                pred2(k) = 0;
            end
        end
        pred = pred2;

        % Keep only labels 1/2 for submission
        keep = pred ~= 0;
        out = [time_ranges(keep,:), pred(keep)];

        % Merge overlapping/adjacent windows into event intervals
        out = mergeIntervals(out, step_s);

        % Export CSV for this trial
        writeCSV(outDir, pid, trialName, out);

        fprintf("%s %s -> %d events\n", pid, trialName, size(out,1));
    end
end

fprintf("Done. CSVs saved in folder: %s\n", outDir);

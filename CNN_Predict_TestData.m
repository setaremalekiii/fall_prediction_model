% Generates CSV predictions for GrandChallengeTestData.mat using cnn_model_gc.mat
% Output CSV columns: [start_time_s, end_time_s, label] where label: 1=NearFall, 2=Fall

clear; clc;

%% Load
T = load("GrandChallengeTestData.mat","test_data");
test_data = T.test_data;

M = load("cnn_model_gc.mat","net","USE_ECG","USE_GSS","winLen_s","Fs","imu_union");
net      = M.net;
USE_ECG  = M.USE_ECG;
USE_GSS  = M.USE_GSS;
winLen_s = M.winLen_s;
Fs       = M.Fs;
imu_union = M.imu_union;

%% Params
step_s     = 1.5;    % sliding step (seconds)
confThresh = 0.75;   % confidence gate
gapTol     = 0.05;   % merge tolerance (seconds)
outDir = "CNN_pred_CSVs";
if ~exist(outDir,"dir"), mkdir(outDir); end

winLen_N = round(winLen_s * Fs);

%% Filters
[bIMU,aIMU] = butter(4, 10/(Fs/2), 'low');            % IMU low-pass
[bECG,aECG] = butter(4, [0.5 40]/(Fs/2), 'bandpass'); % ECG band-pass
[bGSS,aGSS] = butter(4, 5/(Fs/2), 'low');             % GSS low-pass

%% Predict
participants = fieldnames(test_data);

for p = 1:numel(participants)
    pid = participants{p};
    trials = fieldnames(test_data.(pid));

    for t = 1:numel(trials)
        trialName = trials{t};
        data_struct = test_data.(pid).(trialName);

        % Choose an IMU timebase that exists in this trial
        fields = fieldnames(data_struct);
        imuAvail = imu_union(ismember(imu_union, fields));
        if isempty(imuAvail), continue; end

        imu0 = data_struct.(imuAvail{1});
        if isempty(imu0) || size(imu0,1) < 2, continue; end

        tvec = imu0(:,1);
        if isempty(tvec) || (tvec(end) - tvec(1)) < winLen_s
            continue;
        end

        start_times = tvec(1):step_s:(tvec(end)-winLen_s);
        if isempty(start_times), continue; end

        Xtest = {};
        time_ranges = zeros(0,2);

        for w = 1:numel(start_times)
            t1 = start_times(w);
            t2 = t1 + winLen_s;

            seq = build_seq(data_struct, t1, t2, imu_union, winLen_N, ...
                            bIMU,aIMU,bECG,aECG,bGSS,aGSS, USE_ECG, USE_GSS);

            if isempty(seq), continue; end
            Xtest{end+1,1} = seq; %#ok<AGROW>
            time_ranges(end+1,:) = [t1 t2]; %#ok<AGROW>
        end

        if isempty(Xtest), continue; end

        scores = predict(net, Xtest);
        [confMax, cls] = max(scores, [], 2);  % cls: 1=ADL,2=NearFall,3=Fall
        pred = cls - 1;                       % 0/1/2

        % Confidence gate -> push low-confidence to ADL
        pred(confMax < confThresh) = 0;

        % Simple persistence: require at least 2 adjacent windows for NF/F
        pred2 = pred;
        for k = 1:numel(pred)
            if pred(k) == 0, continue; end
            left  = (k>1) && (pred(k-1)==pred(k));
            right = (k<numel(pred)) && (pred(k+1)==pred(k));
            if ~(left || right)
                pred2(k) = 0;
            end
        end
        pred = pred2;

        keep = pred ~= 0;
        out = [time_ranges(keep,:), pred(keep)];

        if isempty(out)
            % still save an empty file (optional); here we save empty
            tok = regexp(trialName,'T\d\d[A-Z]?','match','once');
            if isempty(tok), tok = regexp(trialName,'T\d+','match','once'); end
            if isempty(tok), tok = "TXX"; end
            fname = fullfile(outDir, sprintf("%s_%s.csv", pid, tok));
            writematrix([], fname);
            fprintf("%s %s -> 0 events\n", pid, trialName);
            continue;
        end

        out = merge_events(out, gapTol);

        % filename token keeps A/B (e.g., T01A, T08B)
        tok = regexp(trialName,'T\d\d[A-Z]?','match','once');
        if isempty(tok), tok = regexp(trialName,'T\d+','match','once'); end
        if isempty(tok), tok = "TXX"; end

        fname = fullfile(outDir, sprintf("%s_%s.csv", pid, tok));
        writematrix(out, fname); % no header
        fprintf("%s %s -> %d events\n", pid, trialName, size(out,1));
    end
end

fprintf("Done. CSVs saved in %s\n", outDir);

%% LOCAL FUNCTIONS 

function y = safe_filt(b,a,x)
% Always returns a column vector, same length as input x(:).
    x = x(:);
    if numel(x) <= 24 || any(~isfinite(x)) || std(x) < 1e-12
        y = x;
        return;
    end
    y = filtfilt(b,a,x);
end

function sig_out = pad_trunc(sig, N)
% Pad/truncate to length N (column vector).
    sig = sig(:);
    if numel(sig) >= N
        sig_out = sig(1:N);
    else
        sig_out = [sig; zeros(N-numel(sig),1)];
    end
end

function seq = build_seq(data_struct, t1, t2, imu_union, winLen_N, ...
                         bIMU,aIMU,bECG,aECG,bGSS,aGSS, USE_ECG, USE_GSS)
% Returns seq as [C x T] with per-channel z-score normalization.

    fields = fieldnames(data_struct);
    imuAvail = imu_union(ismember(imu_union, fields));

    accMags = [];
    gyrMags = [];

    % IMU magnitudes per available sensor
    for s = 1:numel(imuAvail)
        imu = data_struct.(imuAvail{s});
        if isempty(imu) || size(imu,2) < 7, continue; end

        idx = imu(:,1) >= t1 & imu(:,1) <= t2;
        w = imu(idx,:);
        if isempty(w) || size(w,1) < 50, continue; end

        acc = w(:,2:4);
        gyr = w(:,5:7);

        accF = zeros(size(acc));
        gyrF = zeros(size(gyr));
        for k = 1:3
            accF(:,k) = safe_filt(bIMU,aIMU, acc(:,k));
            gyrF(:,k) = safe_filt(bIMU,aIMU, gyr(:,k));
        end

        acc_mag = sqrt(sum(accF.^2,2));
        gyr_mag = sqrt(sum(gyrF.^2,2));

        accMags = [accMags, pad_trunc(acc_mag, winLen_N)]; %#ok<AGROW>
        gyrMags = [gyrMags, pad_trunc(gyr_mag, winLen_N)]; %#ok<AGROW>
    end

    % Require IMU (otherwise skip)
    if isempty(accMags) && isempty(gyrMags)
        seq = [];
        return;
    end

    % Aggregate across sensors: mean + max (location-robust)
    acc_mean = mean(accMags, 2);
    acc_max  = max(accMags, [], 2);
    gyr_mean = mean(gyrMags, 2);
    gyr_max  = max(gyrMags, [], 2);

    chans = [acc_mean, acc_max, gyr_mean, gyr_max]; % 4 chans

    % ECG
    if USE_ECG
        if isfield(data_struct,'ECG') && ~isempty(data_struct.ECG)
            ecg = data_struct.ECG;
            idx = ecg(:,1) >= t1 & ecg(:,1) <= t2;
            sig = ecg(idx,2);
            sig = pad_trunc(sig, winLen_N);
            sig = safe_filt(bECG,aECG,sig);
        else
            sig = zeros(winLen_N,1);
        end
        chans = [chans, sig]; %#ok<AGROW>
    end

    % GSS
    if USE_GSS
        if isfield(data_struct,'GSS') && ~isempty(data_struct.GSS)
            gss = data_struct.GSS;
            idx = gss(:,1) >= t1 & gss(:,1) <= t2;
            sig = gss(idx,2);
            sig = pad_trunc(sig, winLen_N);
            sig = safe_filt(bGSS,aGSS,sig);
        else
            sig = zeros(winLen_N,1);
        end
        chans = [chans, sig]; %#ok<AGROW>
    end

    % [C x T]
    seq = chans.'; % C x T

    % Per-channel normalization (match training)
    mu = mean(seq, 2);
    sd = std(seq, 0, 2);
    sd(sd < 1e-6) = 1;
    seq = (seq - mu) ./ sd;
end

function merged = merge_events(events, gapTol)
% events: [start end label], label 1/2. Merges same-label overlaps/gaps.
% If overlap with different labels, keep higher label (Fall=2 overrides NearFall=1).
    if isempty(events)
        merged = [];
        return;
    end

    events = sortrows(events,1);
    merged = events(1,:);

    for i = 2:size(events,1)
        cur = merged(end,:);
        nxt = events(i,:);

        overlaps = nxt(1) <= (cur(2) + gapTol);

        if overlaps
            % If same label -> extend interval
            if nxt(3) == cur(3)
                merged(end,2) = max(cur(2), nxt(2));
            else
                % Different labels overlapping: split by keeping higher label
                hi = max(cur(3), nxt(3)); % 2 overrides 1
                merged(end,3) = hi;
                merged(end,2) = max(cur(2), nxt(2));
            end
        else
            merged(end+1,:) = nxt; %#ok<AGROW>
        end
    end
end

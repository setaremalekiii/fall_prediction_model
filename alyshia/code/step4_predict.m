%% Step 4: Predict on Test Data
% Extracts 459 base features, computes temporal + LOS features,
% predicts with Mdl_final, applies bias + post-processing.
% Writes per-trial CSVs: [start_time, end_time, label] (1=NF, 2=Fall)

addpath('utils');
fprintf('=== STEP 4: Predict on Test Data ===\n');

if ~exist('Fs','var'), Fs = 1000; end
window_sec = 2; overlap_pct = 50;
window_len = window_sec * Fs;
hop_len = floor(window_len * (1 - overlap_pct/100));

%% Load and filter test data
test_file = load('../Data/GrandChallengeTestData.mat');
fn = fieldnames(test_file);
test_struct = [];
for i = 1:length(fn)
    if isstruct(test_file.(fn{i})), test_struct = test_file.(fn{i}); break; end
end

[b_imu,a_imu] = butter(4, 10/(Fs/2), 'low');
[b_ecg,a_ecg] = butter(2, [0.5 40]/(Fs/2), 'bandpass');
[b_gss,a_gss] = butter(4, 5/(Fs/2), 'low');
imu_sensors = {'Back','Left_Thigh','Right_Thigh'};

% Sensor remapping
remap = struct('Sternum','Back','Left_Arm','Left_Thigh','Right_Arm','Right_Thigh');
remap_from = fieldnames(remap);

test_filtered = struct();
test_pids = fieldnames(test_struct);
for p = 1:length(test_pids)
    pID = test_pids{p};
    trials = fieldnames(test_struct.(pID));
    for t = 1:length(trials)
        tID = trials{t};
        sf = fieldnames(test_struct.(pID).(tID));
        for r = 1:length(remap_from)
            src = remap_from{r};
            if isfield(test_struct.(pID).(tID), src)
                test_struct.(pID).(tID).(remap.(src)) = test_struct.(pID).(tID).(src);
                test_struct.(pID).(tID) = rmfield(test_struct.(pID).(tID), src);
            end
        end
        sf = fieldnames(test_struct.(pID).(tID));
        for s = 1:length(sf)
            sID = sf{s}; raw = test_struct.(pID).(tID).(sID);
            if isempty(raw), test_filtered.(pID).(tID).(sID)=raw; continue; end
            tc = raw(:,1);
            if ismember(sID,imu_sensors)
                test_filtered.(pID).(tID).(sID) = [tc, filtfilt(b_imu,a_imu,raw(:,2:end))];
            elseif strcmp(sID,'ECG')
                test_filtered.(pID).(tID).(sID) = [tc, filtfilt(b_ecg,a_ecg,raw(:,2))];
            elseif strcmp(sID,'GSS')
                test_filtered.(pID).(tID).(sID) = [tc, filtfilt(b_gss,a_gss,raw(:,2))];
            else
                test_filtered.(pID).(tID).(sID) = raw;
            end
        end
    end
end
fprintf('  Test data filtered.\n');

%% Predict
n_key = length(temporal_key_idx);
n_los = length(los_feat_idx);
output_dir = 'predictions';
if ~exist(output_dir,'dir'), mkdir(output_dir); end

total_events=0; total_falls=0; total_nf=0;
summary_rows = {};

for p = 1:length(test_pids)
    pID = test_pids{p};
    trials = fieldnames(test_filtered.(pID));
    for t = 1:length(trials)
        tID = trials{t};
        trial = test_filtered.(pID).(tID);

        tok = regexp(tID, '^(T\d+[A-Z]?)', 'tokens');
        if ~isempty(tok), trial_num=tok{1}{1}; else, trial_num=tID; end
        out_name = sprintf('%s_%s', pID, trial_num);
        tok_act = regexp(tID, '_(.+)$', 'tokens');
        if ~isempty(tok_act), activity=tok_act{1}{1}; else, activity=tID; end

        if ~isfield(trial,'Back') || isempty(trial.Back)
            writematrix([], fullfile(output_dir,[out_name '.csv']));
            summary_rows{end+1,1}=pID; summary_rows{end,2}=trial_num;
            summary_rows{end,3}=activity; summary_rows{end,4}=0;
            summary_rows{end,5}=0; summary_rows{end,6}=0;
            summary_rows{end,7}=0; summary_rows{end,8}=0;
            continue;
        end

        n_samples = size(trial.Back,1);
        duration_s = round(n_samples/Fs, 1);
        if n_samples < window_len
            writematrix([], fullfile(output_dir,[out_name '.csv']));
            summary_rows{end+1,1}=pID; summary_rows{end,2}=trial_num;
            summary_rows{end,3}=activity; summary_rows{end,4}=duration_s;
            summary_rows{end,5}=0; summary_rows{end,6}=0;
            summary_rows{end,7}=0; summary_rows{end,8}=0;
            continue;
        end

        time_vec = trial.Back(:,1);
        n_win = floor((n_samples-window_len)/hop_len)+1;

        % Extract base features
        X_base = zeros(n_win, 459);
        w_starts = zeros(n_win,1); w_ends = zeros(n_win,1);
        ws_state = warning('off','signal:findpeaks:largeMinPeakHeight');
        for w = 1:n_win
            ws=(w-1)*hop_len+1; we=ws+window_len-1;
            w_starts(w)=ws; w_ends(w)=we;
            X_base(w,:) = extract_all_features(trial, ws, we, Fs);
        end
        warning(ws_state);
        X_base(isnan(X_base))=0; X_base(isinf(X_base))=0;

        % Temporal features
        Xk = X_base(:, temporal_key_idx);
        X_temp = zeros(n_win, 8*n_key);
        for w = 1:n_win
            curr=Xk(w,:);
            if w>=2, d1=curr-Xk(w-1,:); else, d1=zeros(1,n_key); end
            if w>=3, d2=(curr-Xk(w-1,:))-(Xk(w-1,:)-Xk(w-2,:));
            else, d2=zeros(1,n_key); end
            ws3=max(1,w-2); p3=Xk(ws3:w,:);
            rm3=mean(p3,1); rs3=std(p3,0,1);
            ws5=max(1,w-4); rm5=mean(Xk(ws5:w,:),1);
            ratio=curr./max(abs(rm3),1e-6);
            ws7=max(1,w-6); rs7=std(Xk(ws7:w,:),0,1);
            ws10=max(1,w-9); rs10=std(Xk(ws10:w,:),0,1);
            X_temp(w,:)=[d1,d2,rm3,rs3,rm5,ratio,rs7,rs10];
        end
        X_temp(isnan(X_temp))=0; X_temp(isinf(X_temp))=0;

        % LOS CUSUM
        Xl = X_base(:, los_feat_idx);
        X_cs = zeros(n_win, 3*n_los);
        n_bl=min(5,n_win); bl=mean(Xl(1:n_bl,:),1);
        cp=zeros(1,n_los); cn=zeros(1,n_los);
        for w=1:n_win
            dev=Xl(w,:)-bl;
            cp=max(0,cp+dev); cn=min(0,cn+dev);
            X_cs(w,:)=[cp,cn,cp+abs(cn)];
            cp=cp*0.97; cn=cn*0.97;
        end
        X_cs(isnan(X_cs))=0; X_cs(isinf(X_cs))=0;

        % LOS trend
        X_tr = zeros(n_win, 3*n_los);
        for w=1:n_win
            ws5=max(1,w-4); ws10=max(1,w-9); ws15=max(1,w-14);
            X_tr(w,:)=[local_slope(Xl(ws5:w,:)),local_slope(Xl(ws10:w,:)),local_slope(Xl(ws15:w,:))];
        end
        X_tr(isnan(X_tr))=0; X_tr(isinf(X_tr))=0;

        % Assemble, normalize, predict
        X_test = [X_base, X_temp, X_cs, X_tr];
        X_test_n = (X_test - feat_mean) ./ feat_std;
        [~,prob] = predict(Mdl_final, X_test_n(:, sel_feat));
        prob(:,2)=prob(:,2)+final_bias(2); prob(:,3)=prob(:,3)+final_bias(3);
        [~,pi]=max(prob,[],2); pred=pi-1;

        % Median filter
        if n_win>=5, pred=round(medfilt1(double(pred),5)); end

        % Event-level duration filter (gap=0)
        ii=1;
        while ii<=n_win
            if pred(ii)~=0
                jj=ii+1; while jj<=n_win&&pred(jj)~=0, jj=jj+1; end
                el=jj-ii; elb=pred(ii:jj-1); evt=mode(elb(elb~=0));
                if (evt==1&&el<2)||(evt==2&&el<3), pred(ii:jj-1)=0;
                else, pred(ii:jj-1)=evt; end
                ii=jj;
            else, ii=ii+1; end
        end

        % Convert to events
        trial_results = [];
        ii=1;
        while ii<=n_win
            if pred(ii)~=0
                jj=ii+1; while jj<=n_win&&pred(jj)~=0, jj=jj+1; end
                elb=pred(ii:jj-1); evt=mode(elb(elb~=0));
                s_sec=time_vec(w_starts(ii)); e_sec=time_vec(min(w_ends(jj-1),length(time_vec)));
                trial_results=[trial_results; s_sec, e_sec, evt];
                ii=jj;
            else, ii=ii+1; end
        end

        writematrix(trial_results, fullfile(output_dir,[out_name '.csv']));

        ne=size(trial_results,1);
        nf_t=0; nfl_t=0;
        if ne>0, nf_t=sum(trial_results(:,3)==2); nfl_t=sum(trial_results(:,3)==1); end
        total_events=total_events+ne; total_falls=total_falls+nf_t; total_nf=total_nf+nfl_t;

        summary_rows{end+1,1}=pID; summary_rows{end,2}=trial_num;
        summary_rows{end,3}=activity; summary_rows{end,4}=duration_s;
        summary_rows{end,5}=n_win; summary_rows{end,6}=ne;
        summary_rows{end,7}=nf_t; summary_rows{end,8}=nfl_t;

        fprintf('  %s/%s -> %s.csv: %d win, %d events (F=%d, NF=%d)\n', ...
            pID, tID, out_name, n_win, ne, nf_t, nfl_t);
    end
end

fprintf('\nTotal: %d events (Falls=%d, NearFalls=%d)\n', total_events, total_falls, total_nf);

%% Summary
adl_acts = {'Stand','Pick','Walk','Jog','Lie','Sit','Stairs','JJ'};
n_rows = size(summary_rows,1);
for r=1:n_rows
    is_adl = ismember(summary_rows{r,3}, adl_acts);
    summary_rows{r,9} = is_adl;
    summary_rows{r,10} = is_adl && (summary_rows{r,7}>0||summary_rows{r,8}>0);
    summary_rows{r,11} = ~is_adl && summary_rows{r,6}==0;
end
tbl = cell2table(summary_rows, 'VariableNames', ...
    {'Participant','Trial','Activity','Duration_s','Windows','Events','Falls','NearFalls','IsADL','FP','Missed'});
sf = fullfile(output_dir,'prediction_summary.xlsx');
if exist(sf,'file'), delete(sf); end
writetable(tbl, sf);
fprintf('Summary: %s\nStep 4 done.\n', sf);

%% Helpers
function slopes = local_slope(data)
    n=size(data,1);
    if n<2, slopes=zeros(1,size(data,2)); return; end
    x=(1:n)'-mean(1:n); slopes=(x'*data)/(x'*x);
end

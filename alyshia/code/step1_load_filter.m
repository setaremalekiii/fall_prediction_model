%% Step 1: Load and Filter Data
% Loads raw data, applies Butterworth filters, parses labels.
% Outputs: filtered_data, clean_labels, segments, Fs

fprintf('=== STEP 1: Load and Filter Data ===\n');
Fs = 1000;

if exist('filtered_cache.mat', 'file')
    fprintf('Loading from cache...\n');
    load('filtered_cache.mat', 'filtered_data', 'clean_labels', 'segments');
    fprintf('Step 1 done (cached).\n\n');
    return;
end

load('../Data/GrandChallengeData.mat');

[b_imu, a_imu] = butter(4, 10/(Fs/2), 'low');
[b_ecg, a_ecg] = butter(2, [0.5 40]/(Fs/2), 'bandpass');
[b_gss, a_gss] = butter(4, 5/(Fs/2), 'low');
imu_sensors = {'Back', 'Left_Thigh', 'Right_Thigh'};

filtered_data = struct();
participants = fieldnames(all_data);
for p = 1:length(participants)
    pID = participants{p};
    trials = fieldnames(all_data.(pID));
    fprintf('  %s (%d/%d)\n', pID, p, length(participants));
    for t = 1:length(trials)
        tID = trials{t};
        sensor_fields = fieldnames(all_data.(pID).(tID));
        for s = 1:length(sensor_fields)
            sID = sensor_fields{s};
            raw = all_data.(pID).(tID).(sID);
            if isempty(raw)
                filtered_data.(pID).(tID).(sID) = raw;
                continue;
            end
            tc = raw(:,1);
            if ismember(sID, imu_sensors)
                filtered_data.(pID).(tID).(sID) = [tc, filtfilt(b_imu, a_imu, raw(:,2:end))];
            elseif strcmp(sID, 'ECG')
                filtered_data.(pID).(tID).(sID) = [tc, filtfilt(b_ecg, a_ecg, raw(:,2))];
            elseif strcmp(sID, 'GSS')
                filtered_data.(pID).(tID).(sID) = [tc, filtfilt(b_gss, a_gss, raw(:,2))];
            else
                filtered_data.(pID).(tID).(sID) = raw;
            end
        end
    end
end

%% Parse labels
segments = struct();
label_participants = fieldnames(clean_labels);
for p = 1:length(label_participants)
    pID = label_participants{p};
    label_fields = fieldnames(clean_labels.(pID));
    for lf = 1:length(label_fields)
        lfID = label_fields{lf};
        labels = clean_labels.(pID).(lfID);
        if isempty(labels), continue; end
        nearfall_list = []; fall_list = [];
        i = 1;
        while i <= size(labels, 1)
            if labels(i,1) == 1
                event = struct('movement_start',labels(i,2),'nearfall_start',NaN,...
                    'nearfall_end',NaN,'fall_time',NaN,'end_time',NaN,'outcome','');
                j = i + 1;
                while j <= size(labels,1) && labels(j,1) ~= 1
                    lt=labels(j,1); tm=labels(j,2);
                    if lt==2, event.nearfall_start=tm;
                    elseif lt==3, event.nearfall_end=tm;
                    elseif lt==4, event.fall_time=tm;
                    elseif lt==5, event.end_time=tm; end
                    j = j+1;
                end
                if ~isnan(event.fall_time), event.outcome='fell'; fall_list=[fall_list,event];
                else, event.outcome='recovered'; nearfall_list=[nearfall_list,event]; end
                i = j;
            else
                i = i+1;
            end
        end
        segments.(pID).(lfID).nearfalls = nearfall_list;
        segments.(pID).(lfID).falls = fall_list;
    end
end

save('filtered_cache.mat', 'filtered_data', 'clean_labels', 'segments', '-v7.3');
fprintf('Step 1 done.\n\n');

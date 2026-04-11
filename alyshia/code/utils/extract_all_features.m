function [f, names] = extract_all_features(trial, w_start, w_end, Fs)
% EXTRACT_ALL_FEATURES  Extract all 459 features from a single window.
%   388 original (step2 + step2c + step2d) + 71 lite (step2_lite..step2e_lite)
%
%   Returns:
%     f     — 1×459 feature vector
%     names — 1×459 cell array of feature names (only computed on first call)

    persistent cached_names;
    dt = 1/Fs;
    f = zeros(1, 459);
    col = 0;

    %% ================================================================
    %  PART 1: 388 Original Features
    %  ================================================================

    % Back IMU: 6 raw channels × 14 features = 84
    back = trial.Back(w_start:w_end, 2:7);
    for ch = 1:6
        f(col+1:col+14) = feature_extract_591k(back(:,ch), Fs);
        col = col + 14;
    end

    % Back accel magnitude: 14 features
    back_acc_mag = sqrt(back(:,1).^2 + back(:,2).^2 + back(:,3).^2);
    f(col+1:col+14) = feature_extract_591k(back_acc_mag, Fs);
    col = col + 14;

    % Back angular acceleration (deriv of gyro): 3 × 14 = 42
    for ch = 4:6
        ang_acc = stencil_deriv(dt, back(:,ch));
        f(col+1:col+14) = feature_extract_591k(ang_acc, Fs);
        col = col + 14;
    end

    % Left Thigh: 6 channels + magnitude = 7 × 14 = 98
    has_lt = isfield(trial,'Left_Thigh') && ~isempty(trial.Left_Thigh) && size(trial.Left_Thigh,1) >= w_end;
    if has_lt
        lt = trial.Left_Thigh(w_start:w_end, 2:7);
        for ch = 1:6
            f(col+1:col+14) = feature_extract_591k(lt(:,ch), Fs);
            col = col + 14;
        end
        lt_acc_mag = sqrt(lt(:,1).^2 + lt(:,2).^2 + lt(:,3).^2);
        f(col+1:col+14) = feature_extract_591k(lt_acc_mag, Fs);
        col = col + 14;
    else
        col = col + 98;
    end

    % Right Thigh: 7 × 14 = 98
    has_rt = isfield(trial,'Right_Thigh') && ~isempty(trial.Right_Thigh) && size(trial.Right_Thigh,1) >= w_end;
    if has_rt
        rt = trial.Right_Thigh(w_start:w_end, 2:7);
        for ch = 1:6
            f(col+1:col+14) = feature_extract_591k(rt(:,ch), Fs);
            col = col + 14;
        end
        rt_acc_mag = sqrt(rt(:,1).^2 + rt(:,2).^2 + rt(:,3).^2);
        f(col+1:col+14) = feature_extract_591k(rt_acc_mag, Fs);
        col = col + 14;
    else
        col = col + 98;
    end

    % ECG: 14
    if isfield(trial,'ECG') && ~isempty(trial.ECG) && size(trial.ECG,1) >= w_end
        f(col+1:col+14) = feature_extract_591k(trial.ECG(w_start:w_end, 2), Fs);
    end
    col = col + 14;

    % GSS: 14 + 2 spike features = 16
    has_gss = isfield(trial,'GSS') && ~isempty(trial.GSS) && size(trial.GSS,1) >= w_end;
    if has_gss
        gss_data = trial.GSS(w_start:w_end, 2);
        f(col+1:col+14) = feature_extract_591k(gss_data, Fs);
        f(col+15) = max(abs(gss_data));
        f(col+16) = max(gss_data) - min(gss_data);
    end
    col = col + 16;
    % Total so far: 84+14+42+98+98+14+16 = 366

    % --- Step2c discriminative features (10) ---
    back_acc = back(:,1:3);
    back_gyro = back(:,4:6);
    mean_acc = mean(back_acc, 1);
    f(col+1) = atan2d(norm(mean_acc(1:2)), abs(mean_acc(3)));

    lag_lo = round(0.3*Fs); lag_hi = round(1.5*Fs);
    sig_zm = back_acc_mag - mean(back_acc_mag);
    [acf_b, lags_b] = xcorr(sig_zm, lag_hi, 'coeff');
    acf_pos = acf_b(lags_b >= lag_lo & lags_b <= lag_hi);
    if ~isempty(acf_pos), f(col+2) = max(acf_pos); end

    thresh = mean(back_acc_mag) + 2*std(back_acc_mag);
    n_peaks_val = 0;
    if thresh < max(back_acc_mag)
        ws = warning('off','signal:findpeaks:largeMinPeakHeight');
        [pks,~] = findpeaks(back_acc_mag,'MinPeakHeight',thresh,'MinPeakDistance',round(0.1*Fs));
        warning(ws);
        n_peaks_val = length(pks);
        f(col+3) = n_peaks_val;
    end

    [~, mx_idx] = max(back_acc_mag);
    f(col+4) = mx_idx / Fs;

    mid_pt = floor(length(back_acc_mag)/2);
    f(col+5) = var(back_acc_mag(1:mid_pt)) / max(var(back_acc_mag(mid_pt+1:end)), 1e-6);

    if has_lt
        if ~exist('lt_acc_mag','var')
            lt_a = trial.Left_Thigh(w_start:w_end, 2:4);
            lt_acc_mag = sqrt(lt_a(:,1).^2+lt_a(:,2).^2+lt_a(:,3).^2);
        end
        [acf_lt,lags_lt] = xcorr(lt_acc_mag-mean(lt_acc_mag), lag_hi, 'coeff');
        ap = acf_lt(lags_lt>=lag_lo & lags_lt<=lag_hi);
        if ~isempty(ap), f(col+6) = max(ap); end
    end

    if has_rt
        if ~exist('rt_acc_mag','var')
            rt_a = trial.Right_Thigh(w_start:w_end, 2:4);
            rt_acc_mag = sqrt(rt_a(:,1).^2+rt_a(:,2).^2+rt_a(:,3).^2);
        end
        [acf_rt,lags_rt] = xcorr(rt_acc_mag-mean(rt_acc_mag), lag_hi, 'coeff');
        ap = acf_rt(lags_rt>=lag_lo & lags_rt<=lag_hi);
        if ~isempty(ap), f(col+7) = max(ap); end
    end

    back_gyro_mag = sqrt(back_gyro(:,1).^2+back_gyro(:,2).^2+back_gyro(:,3).^2);
    f(col+8) = max(back_gyro_mag);
    col = col + 8;
    % Total: 374

    % --- Step2d biomechanical features (14) ---
    jerk = diff(back_acc_mag) * Fs;
    f(col+1) = max(abs(jerk));
    f(col+2) = mean(abs(jerk));
    f(col+3) = sum(jerk.^2)/length(jerk);
    f(col+4) = mean(abs(back_acc(:,1))+abs(back_acc(:,2))+abs(back_acc(:,3)));

    if has_lt
        r = corrcoef(back_acc_mag, lt_acc_mag); f(col+5) = r(1,2);
    end
    if has_rt
        r = corrcoef(back_acc_mag, rt_acc_mag); f(col+6) = r(1,2);
    end
    if has_lt && has_rt
        r = corrcoef(lt_acc_mag, rt_acc_mag); f(col+7) = r(1,2);
    end

    n_samp = length(back_acc_mag);
    q4_start = floor(3*n_samp/4)+1;
    var_q4 = var(back_acc_mag(q4_start:end));
    var_full = var(back_acc_mag);
    f(col+8) = var_q4;
    f(col+9) = var_q4/max(var_full,1e-6);

    zm = back_acc_mag - mean(back_acc_mag);
    f(col+10) = sum(abs(diff(sign(zm)))>0)/length(zm);
    f(col+11) = sqrt(mean(back_acc_mag.^2));
    f(col+12) = sqrt(mean(back_gyro_mag.^2));
    f(col+13) = iqr(back_acc_mag);
    f(col+14) = skewness(back_acc_mag);
    col = col + 14;
    % Total so far: 366+8+14 = 388

    %% ================================================================
    %  PART 2: 71 Lite Features (step2_lite + step2c_lite + step2d_lite + step2e_lite)
    %  ================================================================
    L = length(back_acc_mag);

    % --- step2_lite: 32 features ---
    f(col+1) = max(back_acc_mag);
    f(col+2) = min(back_acc_mag);
    f(col+3) = std(back_acc_mag);
    f(col+4) = kurtosis(back_acc_mag);
    f(col+5) = sqrt(mean(back_acc_mag.^2));
    f(col+6) = iqr(back_acc_mag);
    f(col+7) = skewness(back_acc_mag);

    Y_fft = fft(back_acc_mag - mean(back_acc_mag));
    P2_f = abs(Y_fft/L);
    P1_f = P2_f(1:floor(L/2)+1);
    P1_f(2:end-1) = 2*P1_f(2:end-1);
    freq = Fs*(0:floor(L/2))/L;
    [~,idx_dom] = max(P1_f);
    f(col+8) = freq(idx_dom);
    f(col+9) = trapz(P1_f);

    f(col+10) = max(back_gyro_mag);
    f(col+11) = sqrt(mean(back_gyro_mag.^2));
    f(col+12) = std(back_gyro_mag);

    f(col+13) = max(abs(jerk));
    f(col+14) = mean(abs(jerk));
    f(col+15) = sum(jerk.^2)/length(jerk);

    f(col+16) = mean(abs(back_acc(:,1))+abs(back_acc(:,2))+abs(back_acc(:,3)));
    f(col+17) = atan2d(norm(mean_acc(1:2)), abs(mean_acc(3)));

    f(col+18) = var(back_acc_mag(1:mid_pt))/max(var(back_acc_mag(mid_pt+1:end)),1e-6);
    f(col+19) = var_q4/max(var_full,1e-6);
    f(col+20) = sum(abs(diff(sign(zm)))>0)/L;

    f(col+21) = n_peaks_val;
    f(col+22) = mx_idx/Fs;

    for ax = 1:3
        f(col+22+(ax-1)*2+1) = max(abs(back_acc(:,ax)));
        f(col+22+(ax-1)*2+2) = std(back_acc(:,ax));
    end

    if has_lt && has_rt
        r = corrcoef(lt_acc_mag, rt_acc_mag); f(col+29) = r(1,2);
    end
    if ~isempty(acf_pos), f(col+30) = max(acf_pos); end

    if has_gss
        f(col+31) = max(abs(gss_data));
        f(col+32) = max(gss_data)-min(gss_data);
    end
    col = col + 32;

    % --- step2c_lite: 17 features ---
    if has_lt, f(col+1)=max(lt_acc_mag); f(col+2)=std(lt_acc_mag); f(col+3)=sqrt(mean(lt_acc_mag.^2)); end
    if has_rt, f(col+4)=max(rt_acc_mag); f(col+5)=std(rt_acc_mag); f(col+6)=sqrt(mean(rt_acc_mag.^2)); end
    if has_lt, r=corrcoef(back_acc_mag,lt_acc_mag); f(col+7)=r(1,2); end
    if has_rt, r=corrcoef(back_acc_mag,rt_acc_mag); f(col+8)=r(1,2); end

    accum = cumtrapz(P1_f);
    idx_10=find(freq>=10,1,'first'); idx_20=find(freq>=20,1,'first');
    idx_1=find(freq>=1,1,'first'); idx_3=find(freq>=3,1,'first');
    if ~isempty(idx_10), f(col+9)=accum(idx_10-1); end
    if ~isempty(idx_10)&&~isempty(idx_20), f(col+10)=accum(idx_20-1)-accum(idx_10-1); end
    if ~isempty(idx_1)&&~isempty(idx_3), f(col+11)=accum(idx_3-1)-accum(idx_1-1); end

    energy_total = sum(back_acc_mag.^2);
    energy_q4 = sum(back_acc_mag(q4_start:end).^2);
    f(col+12) = energy_q4/max(energy_total,1e-9);
    q2_s=floor(L/4)+1; q2_e=floor(L/2);
    std_q4=std(back_acc_mag(q4_start:end)); std_q2=std(back_acc_mag(q2_s:q2_e));
    f(col+13) = std_q4/max(std_q2,1e-9);

    f(col+14) = mean(abs(back_gyro_mag));
    gyro_jerk = diff(back_gyro_mag)*Fs;
    f(col+15) = max(abs(gyro_jerk));

    energy_h1=sum(back_acc_mag(1:mid_pt).^2); energy_h2=sum(back_acc_mag(mid_pt+1:end).^2);
    f(col+16) = energy_h2/max(energy_h1,1e-9);
    f(col+17) = max(back_acc_mag)/max(mean(back_acc_mag),1e-9);
    col = col + 17;

    % --- step2d_lite: 14 sub-window features ---
    sw_len=500; sw_hop=125;
    n_sw = floor((L-sw_len)/sw_hop)+1;
    if n_sw >= 1
        delta_a = zeros(n_sw,1);
        for s=1:n_sw
            ss=(s-1)*sw_hop+1; se=ss+sw_len-1;
            sw_a=back_acc(ss:se,:);
            dx=max(sw_a(:,1))-min(sw_a(:,1));
            dy=max(sw_a(:,2))-min(sw_a(:,2));
            dz=max(sw_a(:,3))-min(sw_a(:,3));
            delta_a(s)=sqrt(dx^2+dy^2+dz^2);
        end
        [~,best_sw]=max(delta_a);
        sw_s=(best_sw-1)*sw_hop+1; sw_e=sw_s+sw_len-1;
        sw_acc=back_acc(sw_s:sw_e,:); sw_gyro=back_gyro(sw_s:sw_e,:);
        sw_acc_mag=sqrt(sw_acc(:,1).^2+sw_acc(:,2).^2+sw_acc(:,3).^2);
        sw_gyro_mag=sqrt(sw_gyro(:,1).^2+sw_gyro(:,2).^2+sw_gyro(:,3).^2);

        f(col+1)=delta_a(best_sw);
        f(col+2)=mean(sw_acc_mag); f(col+3)=std(sw_acc_mag);
        f(col+4)=mean(abs(diff(sw_acc_mag)));
        dot_p=sum(sw_acc(1:end-1,:).*sw_acc(2:end,:),2);
        np=sw_acc_mag(1:end-1).*sw_acc_mag(2:end);
        ca=dot_p./max(np,1e-9); ca=max(-1,min(1,ca));
        f(col+5)=mean(acosd(ca));
        hz_mag=sqrt(sw_acc(:,1).^2+sw_acc(:,2).^2);
        f(col+6)=mean(hz_mag);
        f(col+7)=max(sw_acc_mag); f(col+8)=min(sw_acc_mag);
        f(col+9)=kurtosis(sw_acc_mag);
        sw_jerk=diff(sw_acc_mag)*Fs;
        f(col+10)=max(abs(sw_jerk));
        msw=mean(sw_acc,1);
        f(col+11)=atan2d(norm(msw(1:2)),abs(msw(3)));
        f(col+12)=max(sw_gyro_mag); f(col+13)=mean(sw_gyro_mag);
        f(col+14)=(sw_s-1)/max(L-sw_len,1);
    end
    col = col + 14;

    % --- step2e_lite: 8 spectral/entropy features ---
    P1_norm = P1_f/max(sum(P1_f),1e-9);
    P1_norm(P1_norm==0)=1e-12;
    f(col+1) = -sum(P1_norm.*log2(P1_norm))/log2(length(P1_norm));

    total_pow = sum(P1_f);
    if total_pow > 1e-9
        f(col+2) = sum(freq(:).*P1_f(:))/total_pow;
        f(col+3) = sqrt(sum(P1_f(:).*(freq(:)-f(col+2)).^2)/total_pow);
    end

    [counts_h,~]=histcounts(back_acc_mag,20);
    prob_h=counts_h/sum(counts_h); prob_h(prob_h==0)=[];
    f(col+4) = -sum(prob_h.*log2(prob_h));

    for ax=1:3
        aa=diff(back_gyro(:,ax))*Fs;
        f(col+4+ax)=sqrt(mean(aa.^2));
    end
    aa_x=diff(back_gyro(:,1))*Fs; aa_y=diff(back_gyro(:,2))*Fs; aa_z=diff(back_gyro(:,3))*Fs;
    f(col+8)=max(sqrt(aa_x.^2+aa_y.^2+aa_z.^2));
    col = col + 8;

    % Clean
    f(isnan(f))=0; f(isinf(f))=0;

    %% ================================================================
    %  Feature names (built once, cached)
    %  ================================================================
    if nargout > 1
        if isempty(cached_names)
            cached_names = build_names();
        end
        names = cached_names;
    end
end

%% ====================================================================
function names = build_names()
    base14 = {'Mean','Max','Min','Std','Kurtosis','DomFreq','2ndPeak','3rdPeak','PeakRatio','MeanFreq','TotalPow','LowPow','HighPow','PhysPow'};
    names = {};

    back_ch = {'BackAccX','BackAccY','BackAccZ','BackGyroX','BackGyroY','BackGyroZ'};
    for c=1:6, for ff=1:14, names{end+1}=[back_ch{c} '_' base14{ff}]; end, end
    for ff=1:14, names{end+1}=['BackAccMag_' base14{ff}]; end
    ang_ch = {'BackAngAccX','BackAngAccY','BackAngAccZ'};
    for c=1:3, for ff=1:14, names{end+1}=[ang_ch{c} '_' base14{ff}]; end, end
    lt_ch = {'LTAccX','LTAccY','LTAccZ','LTGyroX','LTGyroY','LTGyroZ'};
    for c=1:6, for ff=1:14, names{end+1}=[lt_ch{c} '_' base14{ff}]; end, end
    for ff=1:14, names{end+1}=['LTAccMag_' base14{ff}]; end
    rt_ch = {'RTAccX','RTAccY','RTAccZ','RTGyroX','RTGyroY','RTGyroZ'};
    for c=1:6, for ff=1:14, names{end+1}=[rt_ch{c} '_' base14{ff}]; end, end
    for ff=1:14, names{end+1}=['RTAccMag_' base14{ff}]; end
    for ff=1:14, names{end+1}=['ECG_' base14{ff}]; end
    for ff=1:14, names{end+1}=['GSS_' base14{ff}]; end
    names = [names, {'GSS_PeakAbs','GSS_Range'}];

    names = [names, {'BackTiltAngle','BackAccMag_Periodicity','BackAccMag_NPeaks', ...
        'BackAccMag_RiseTime','BackAccMag_VarRatio', ...
        'LTAccMag_Periodicity','RTAccMag_Periodicity','BackGyroMag_Max'}];

    names = [names, {'BackAccMag_MaxJerk','BackAccMag_MeanAbsJerk','BackAccMag_JerkEnergy', ...
        'Back_SMA','BackLT_AccMagCorr','BackRT_AccMagCorr','LTRT_AccMagCorr', ...
        'BackAccMag_Q4Var','BackAccMag_Q4VarRatio','BackAccMag_ZCR', ...
        'BackAccMag_RMS','BackGyroMag_RMS','BackAccMag_IQR','BackAccMag_Skewness'}];

    lite_names = { ...
        'LITE_BackAccMag_Max','LITE_BackAccMag_Min','LITE_BackAccMag_Std', ...
        'LITE_BackAccMag_Kurtosis','LITE_BackAccMag_RMS','LITE_BackAccMag_IQR', ...
        'LITE_BackAccMag_Skewness','LITE_BackAccMag_DomFreq','LITE_BackAccMag_TotalPow', ...
        'LITE_BackGyroMag_Max','LITE_BackGyroMag_RMS','LITE_BackGyroMag_Std', ...
        'LITE_BackAccMag_MaxJerk','LITE_BackAccMag_MeanAbsJerk','LITE_BackAccMag_JerkEnergy', ...
        'LITE_Back_SMA','LITE_BackTiltAngle', ...
        'LITE_BackAccMag_PrePostVarRatio','LITE_BackAccMag_Q4VarRatio','LITE_BackAccMag_ZCR', ...
        'LITE_BackAccMag_NPeaks','LITE_BackAccMag_RiseTime', ...
        'LITE_BackAccX_MaxAbs','LITE_BackAccX_Std','LITE_BackAccY_MaxAbs','LITE_BackAccY_Std', ...
        'LITE_BackAccZ_MaxAbs','LITE_BackAccZ_Std', ...
        'LITE_LTRT_AccMagCorr','LITE_BackAccMag_Periodicity', ...
        'LITE_GSS_PeakAbs','LITE_GSS_Range', ...
        'LITE_LTAccMag_Max','LITE_LTAccMag_Std','LITE_LTAccMag_RMS', ...
        'LITE_RTAccMag_Max','LITE_RTAccMag_Std','LITE_RTAccMag_RMS', ...
        'LITE_BackLT_AccMagCorr','LITE_BackRT_AccMagCorr', ...
        'LITE_BackAccMag_LowPow','LITE_BackAccMag_HighPow','LITE_BackAccMag_PhysPow', ...
        'LITE_BackAccMag_Q4EnergyRatio','LITE_BackAccMag_Q4Q2StdRatio', ...
        'LITE_BackGyroMag_MeanAbs','LITE_BackGyroMag_JerkMax', ...
        'LITE_BackAccMag_EnergyShift','LITE_BackAccMag_Impulsiveness', ...
        'LITE_SW_DeltaAccMag','LITE_SW_MeanAccMag','LITE_SW_StdAccMag', ...
        'LITE_SW_MeanAbsDiff','LITE_SW_MeanAngleChange','LITE_SW_MeanHorizMag', ...
        'LITE_SW_MaxAccMag','LITE_SW_MinAccMag','LITE_SW_KurtAccMag', ...
        'LITE_SW_MaxJerk','LITE_SW_TiltAngle', ...
        'LITE_SW_MaxGyroMag','LITE_SW_MeanGyroMag','LITE_SW_Position', ...
        'LITE_BackAccMag_SpectralEntropy','LITE_BackAccMag_SpectralCentroid', ...
        'LITE_BackAccMag_SpectralSpread','LITE_BackAccMag_ShannonEntropy', ...
        'LITE_BackGyroX_AngAccRMS','LITE_BackGyroY_AngAccRMS','LITE_BackGyroZ_AngAccRMS', ...
        'LITE_BackGyroMag_AngAccMax'};
    names = [names, lite_names];
end

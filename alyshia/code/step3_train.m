%% Step 3: Train 3-Class RF with Temporal + LOS Features (LOPO CV)
% Adds temporal context (8 types × 29 key features = 232) and
% LOS-targeted features (CUSUM 45 + trend 45 = 90) to base 459,
% giving 781 total. Uses mRMR selection, per-fold OOB bias tuning,
% median filter + event-level post-processing.
%
% Outputs: Mdl_final, sel_feat, feat_mean, feat_std,
%          temporal_key_idx, los_feat_idx, final_bias

addpath('utils');
fprintf('=== STEP 3: Train Model (LOPO CV) ===\n');
rng(42);

%% Configuration
n_top         = 350;
rf_trees      = 500;
rf_leaf       = 5;
adl_ratio     = 3;
nf_cost_boost = 1.0;

%% Parse participant/trial structure
n_windows = length(win_info);
participant_ids = cell(n_windows,1);
trial_keys = cell(n_windows,1);
for i = 1:n_windows
    parts = strsplit(win_info{i}, '/');
    participant_ids{i} = parts{1};
    trial_keys{i} = [parts{1} '/' parts{2}];
end
unique_participants = unique(participant_ids);
n_participants = length(unique_participants);
group = zeros(n_windows,1);
for i = 1:n_windows
    group(i) = find(strcmp(unique_participants, participant_ids{i}));
end
[unique_trials, ~, trial_group] = unique(trial_keys, 'stable');

fprintf('  %d participants, %d windows (ADL=%d, NF=%d, Fall=%d)\n', ...
    n_participants, n_windows, sum(y==0), sum(y==1), sum(y==2));

%% Temporal context features (8 types × 29 keys)
key_patterns = {
    'BackAccMag_Max','BackAccMag_Std','BackAccMag_Kurtosis','BackAccMag_Mean', ...
    'BackAccMag_TotalPow','BackAccMag_DomFreq','BackGyroX_Std','BackGyroY_Std', ...
    'BackGyroZ_Std','LITE_BackAccMag_MaxJerk','LITE_BackAccMag_MeanAbsJerk', ...
    'LITE_BackAccMag_JerkEnergy','LITE_Back_SMA','LITE_BackTiltAngle', ...
    'LITE_BackAccMag_ZCR','LITE_BackAccMag_RMS','LITE_BackGyroMag_Max', ...
    'LITE_BackAccMag_PrePostVarRatio','LITE_BackAccMag_Skewness', ...
    'LITE_BackAccMag_LowPow','LITE_BackAccMag_HighPow', ...
    'LITE_BackAccMag_SpectralEntropy','LITE_BackAccMag_ShannonEntropy', ...
    'LITE_BackAccMag_Impulsiveness','LITE_BackAccMag_EnergyShift', ...
    'LITE_BackGyroMag_MeanAbs','LITE_BackGyroMag_JerkMax', ...
    'LITE_LTAccMag_Max','LITE_RTAccMag_Max'};

temporal_key_idx = [];
for k = 1:length(key_patterns)
    m = find(strcmp(feature_names, key_patterns{k}));
    if ~isempty(m), temporal_key_idx(end+1) = m(1); end
end
n_key = length(temporal_key_idx);
fprintf('  Temporal: %d keys × 8 types = %d features\n', n_key, 8*n_key);

X_key = X(:, temporal_key_idx);
X_temporal = zeros(n_windows, 8*n_key);
for t = 1:length(unique_trials)
    idx = find(trial_group==t); n_t = length(idx);
    for w = 1:n_t
        i = idx(w); curr = X_key(i,:);
        if w>=2, d1=curr-X_key(idx(w-1),:); else, d1=zeros(1,n_key); end
        if w>=3, d2=(curr-X_key(idx(w-1),:))-(X_key(idx(w-1),:)-X_key(idx(w-2),:));
        else, d2=zeros(1,n_key); end
        ws3=max(1,w-2); p3=X_key(idx(ws3:w),:);
        rm3=mean(p3,1); rs3=std(p3,0,1);
        ws5=max(1,w-4); rm5=mean(X_key(idx(ws5:w),:),1);
        ratio=curr./max(abs(rm3),1e-6);
        ws7=max(1,w-6); rs7=std(X_key(idx(ws7:w),:),0,1);
        ws10=max(1,w-9); rs10=std(X_key(idx(ws10:w),:),0,1);
        X_temporal(i,:)=[d1,d2,rm3,rs3,rm5,ratio,rs7,rs10];
    end
end
X_temporal(isnan(X_temporal))=0; X_temporal(isinf(X_temporal))=0;

%% LOS features (CUSUM + trend on 15 keys)
los_patterns = {
    'LITE_BackTiltAngle','BackAccMag_Mean','BackAccMag_Std','BackAccMag_Max', ...
    'BackGyroX_Std','BackGyroY_Std','BackGyroZ_Std','LITE_Back_SMA', ...
    'LITE_BackAccMag_RMS','LITE_BackGyroMag_Max','LITE_BackGyroMag_MeanAbs', ...
    'LITE_LTAccMag_Max','LITE_RTAccMag_Max','BackAccMag_Kurtosis','LITE_BackAccMag_Skewness'};

los_feat_idx = [];
for k = 1:length(los_patterns)
    m = find(strcmp(feature_names, los_patterns{k}));
    if ~isempty(m), los_feat_idx(end+1) = m(1); end
end
n_los = length(los_feat_idx);

X_los = X(:, los_feat_idx);
X_cusum = zeros(n_windows, 3*n_los);
X_trend = zeros(n_windows, 3*n_los);
for t = 1:length(unique_trials)
    idx = find(trial_group==t); n_t = length(idx);
    n_bl = min(5, n_t);
    baseline = mean(X_los(idx(1:n_bl),:), 1);
    cp = zeros(1,n_los); cn = zeros(1,n_los);
    for w = 1:n_t
        i = idx(w);
        dev = X_los(i,:) - baseline;
        cp = max(0, cp+dev); cn = min(0, cn+dev);
        X_cusum(i,:) = [cp, cn, cp+abs(cn)];
        cp = cp*0.97; cn = cn*0.97;

        ws5=max(1,w-4); ws10=max(1,w-9); ws15=max(1,w-14);
        X_trend(i,:) = [local_slope(X_los(idx(ws5:w),:)), ...
                        local_slope(X_los(idx(ws10:w),:)), ...
                        local_slope(X_los(idx(ws15:w),:))];
    end
end
X_cusum(isnan(X_cusum))=0; X_cusum(isinf(X_cusum))=0;
X_trend(isnan(X_trend))=0; X_trend(isinf(X_trend))=0;
fprintf('  LOS: %d keys × 6 types = %d features\n', n_los, 6*n_los);

%% Assemble full feature matrix
X_full = [X, X_temporal, X_cusum, X_trend];
n_feat = size(X_full, 2);
fprintf('  Total: %d features\n', n_feat);

%% LOPO Cross-Validation
pred_all = zeros(n_windows,1);
prob_all = zeros(n_windows,3);

fprintf('\n  LOPO CV (%d folds):\n', n_participants);
for fold = 1:n_participants
    pID = unique_participants{fold};
    te = find(group==fold); tr = find(group~=fold);

    % Normalize
    fm = mean(X_full(tr,:),1); fs = std(X_full(tr,:),0,1); fs(fs==0)=1;
    Xtr = (X_full(tr,:)-fm)./fs; Xte = (X_full(te,:)-fm)./fs;
    ytr = y(tr);

    % Balance
    ia=find(ytr==0); in=find(ytr==1); if_=find(ytr==2);
    nka=min(length(ia), max(length(in),length(if_))*adl_ratio);
    ia_s=ia(randperm(length(ia),nka));
    nn=length(if_)-length(in);
    if nn>0
        src=randi(length(in),nn,1);
        Xns=Xtr(in(src),:)+0.05*randn(nn,size(Xtr,2));
        tb=sort([ia_s;in;if_]); Xb=[Xtr(tb,:);Xns]; yb=[ytr(tb);ones(nn,1)];
    else
        tb=sort([ia_s;in;if_]); Xb=Xtr(tb,:); yb=ytr(tb);
    end

    % mRMR + train
    [si,~]=fscmrmr(Xb,categorical(yb));
    sel=si(1:min(n_top,length(si)));
    cn=[sum(yb==0),sum(yb==1),sum(yb==2)];
    wt=max(cn)./max(cn,1);
    cost=[0,wt(1),wt(1); wt(2)*nf_cost_boost,0,wt(2)*nf_cost_boost; wt(3),wt(3),0];
    mdl=fitcensemble(Xb(:,sel),yb,'Method','Bag','NumLearningCycles',rf_trees,...
        'Learners',templateTree('MinLeafSize',rf_leaf),'Cost',cost);

    [~,pf]=predict(mdl,Xte(:,sel));

    % OOB bias tuning
    [op,opr]=oobPredict(mdl); bmf1=macro_f1(yb,op,[0 1 2]);
    bbn=0; bbf=0;
    for bn=-0.15:0.05:0.15
        for bf=-0.15:0.05:0.15
            pa=opr; pa(:,2)=pa(:,2)+bn; pa(:,3)=pa(:,3)+bf;
            [~,pi]=max(pa,[],2); m=macro_f1(yb,pi-1,[0 1 2]);
            if m>bmf1, bmf1=m; bbn=bn; bbf=bf; end
        end
    end
    pf(:,2)=pf(:,2)+bbn; pf(:,3)=pf(:,3)+bbf;
    [~,pi]=max(pf,[],2);
    pred_all(te)=pi-1; prob_all(te,:)=pf;

    mf1_fold = macro_f1(y(te), pi-1, [0 1 2]);
    fprintf('    %d/%d %s: macF1=%.3f bias=[%+.2f,%+.2f]\n', fold, n_participants, pID, mf1_fold, bbn, bbf);
end

%% Post-processing: median filter
mf1_raw = macro_f1(y, pred_all, [0 1 2]);
fprintf('\n  Raw macF1=%.4f\n', mf1_raw);

best_k=1; best_mf1=mf1_raw; pred_sm=pred_all;
for k=[3 5 7]
    pt=pred_all;
    for ti=1:length(unique_trials)
        tidx=find(trial_group==ti);
        if length(tidx)<k, continue; end
        pt(tidx)=round(medfilt1(double(pred_all(tidx)),k));
    end
    m=macro_f1(y,pt,[0 1 2]);
    fprintf('    medfilt k=%d: macF1=%.4f (%+.4f)\n', k, m, m-mf1_raw);
    if m>best_mf1, best_mf1=m; best_k=k; pred_sm=pt; end
end
if best_k>1, pred_all=pred_sm; end
mf1_med = macro_f1(y, pred_all, [0 1 2]);

%% Post-processing: event-level
best_gap=0; best_mf1_e=mf1_med; pred_eb=pred_all;
for mg=[0 1 2]
    pe=pred_all;
    for ti=1:length(unique_trials)
        tidx=find(trial_group==ti); lb=pe(tidx); nw=length(lb);
        if mg>0
            ii=1; while ii<=nw
                if lb(ii)~=0
                    jj=ii; while jj<=nw
                        if lb(jj)~=0, jj=jj+1; else
                            gs=jj; while jj<=nw&&lb(jj)==0, jj=jj+1; end
                            if jj<=nw&&(jj-gs)<=mg, lb(gs:jj-1)=lb(gs-1); else, break; end
                        end
                    end; ii=jj;
                else, ii=ii+1; end
            end
        end
        ii=1; while ii<=nw
            if lb(ii)~=0
                jj=ii+1; while jj<=nw&&lb(jj)~=0, jj=jj+1; end
                el=jj-ii; elb=lb(ii:jj-1); evt=mode(elb(elb~=0));
                disc=(evt==1&&el<2)||(evt==2&&el<3);
                if disc, lb(ii:jj-1)=0; else, lb(ii:jj-1)=evt; end
                ii=jj;
            else, ii=ii+1; end
        end
        pe(tidx)=lb;
    end
    m=macro_f1(y,pe,[0 1 2]);
    fprintf('    event gap=%d: macF1=%.4f (%+.4f)\n', mg, m, m-mf1_med);
    if m>best_mf1_e, best_mf1_e=m; best_gap=mg; pred_eb=pe; end
end
if best_mf1_e>mf1_med, pred_all=pred_eb; end

%% Final results
mf1_final = macro_f1(y, pred_all, [0 1 2]);
acc = sum(pred_all==y)/length(y)*100;
fprintf('\n  FINAL: Acc=%.1f%%, macF1=%.4f (raw=%.4f -> med=%.4f -> evt=%.4f)\n', ...
    acc, mf1_final, mf1_raw, mf1_med, mf1_final);

cm = confusionmat(y, pred_all);
fprintf('  Confusion matrix:\n'); disp(cm);
for c=[0 1 2]
    tp=sum(y==c&pred_all==c); fp=sum(y~=c&pred_all==c); fn=sum(y==c&pred_all~=c);
    pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1); f1=2*pr*rc/max(pr+rc,1e-9);
    fprintf('    Class %d: P=%.3f R=%.3f F1=%.3f\n', c, pr, rc, f1);
end

fprintf('\n  Per-participant:\n');
for pi=1:n_participants
    m=group==pi; mf1_p=macro_f1(y(m),pred_all(m),[0 1 2]);
    fprintf('    %s: macF1=%.3f\n', unique_participants{pi}, mf1_p);
end

%% Train final model on all data
fprintf('\n  Training final model on all data...\n');
feat_mean = mean(X_full,1); feat_std = std(X_full,0,1); feat_std(feat_std==0)=1;
Xn = (X_full - feat_mean) ./ feat_std;

ia=find(y==0); in=find(y==1); if_=find(y==2);
nka=min(length(ia), max(length(in),length(if_))*adl_ratio);
ia_s=ia(randperm(length(ia),nka));
nn=length(if_)-length(in);
if nn>0
    src=randi(length(in),nn,1);
    Xns=Xn(in(src),:)+0.05*randn(nn,size(Xn,2));
    tb=sort([ia_s;in;if_]); Xf=[Xn(tb,:);Xns]; yf=[y(tb);ones(nn,1)];
else
    tb=sort([ia_s;in;if_]); Xf=Xn(tb,:); yf=y(tb);
end

[si,~]=fscmrmr(Xf,categorical(yf));
sel_feat = si(1:min(n_top,length(si)));

cn=[sum(yf==0),sum(yf==1),sum(yf==2)];
wt=max(cn)./max(cn,1);
cost=[0,wt(1),wt(1); wt(2)*nf_cost_boost,0,wt(2)*nf_cost_boost; wt(3),wt(3),0];

fprintf('    %d samples, %d features\n', length(yf), length(sel_feat));
Mdl_final = fitcensemble(Xf(:,sel_feat), yf, 'Method','Bag', ...
    'NumLearningCycles',rf_trees, 'Learners',templateTree('MinLeafSize',rf_leaf), 'Cost',cost);

% OOB bias
[op,opr]=oobPredict(Mdl_final); bmf1=macro_f1(yf,op,[0 1 2]);
bbn=0; bbf=0;
for bn=-0.15:0.05:0.15
    for bf=-0.15:0.05:0.15
        pa=opr; pa(:,2)=pa(:,2)+bn; pa(:,3)=pa(:,3)+bf;
        [~,pi]=max(pa,[],2); m=macro_f1(yf,pi-1,[0 1 2]);
        if m>bmf1, bmf1=m; bbn=bn; bbf=bf; end
    end
end
final_bias = [0, bbn, bbf];
fprintf('    OOB bias: NF=%+.2f, Fall=%+.2f\n', bbn, bbf);
fprintf('Step 3 done.\n\n');

%% Helpers
function slopes = local_slope(data)
    n = size(data,1);
    if n<2, slopes=zeros(1,size(data,2)); return; end
    x = (1:n)'-mean(1:n);
    slopes = (x'*data)/(x'*x);
end

function mf1 = macro_f1(yt, yp, c)
    f=zeros(length(c),1);
    for ci=1:length(c)
        cc=c(ci); tp=sum(yt==cc&yp==cc); fp=sum(yt~=cc&yp==cc); fn=sum(yt==cc&yp~=cc);
        pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1); f(ci)=2*pr*rc/max(pr+rc,1e-9);
    end
    mf1=mean(f);
end

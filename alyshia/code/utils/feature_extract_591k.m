function feature_list = feature_extract_591k(signal, Fs)
    feature_list = [time_domain_features(signal), freq_domain_features(signal, Fs)];
end

function feature_list = time_domain_features(signal)
    feature_list(1) = mean(signal);
    feature_list(2) = max(signal);
    feature_list(3) = min(signal);
    feature_list(4) = std(signal);
    feature_list(5) = kurtosis(signal);
end

function feature_list = freq_domain_features(signal, Fs)
    L = length(signal);
    Y = fft(signal - mean(signal));
    P2 = abs(Y/L);
    P1 = P2(1:floor(L/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = Fs*(0:floor(L/2))/L;

    [pks, locs] = findpeaks(P1, 'SortStr', 'descend', 'NPeaks', 3);
    [~, ind] = max(P1);
    accum_fft = cumtrapz(P1);

    feature_list(1) = f(ind);
    if length(locs) > 1, feature_list(2) = f(locs(2)); else, feature_list(2) = 0; end
    if length(locs) > 2, feature_list(3) = f(locs(3)); else, feature_list(3) = 0; end
    if length(locs) > 1, feature_list(4) = pks(1) / pks(2); else, feature_list(4) = 0; end

    feature_list(5) = f(find(accum_fft > accum_fft(end)/2, 1, 'first'));
    feature_list(6) = accum_fft(end);

    tenhz = find(f >= 10, 1, 'first');
    feature_list(7) = accum_fft(tenhz - 1);
    twentyhz = find(f >= 20, 1, 'first');
    feature_list(8) = accum_fft(twentyhz - 1) - accum_fft(tenhz - 1);
    onehz = find(f >= 1, 1, 'first');
    threehz = find(f >= 3, 1, 'first');
    feature_list(9) = accum_fft(threehz - 1) - accum_fft(onehz - 1);
end

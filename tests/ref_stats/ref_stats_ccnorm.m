% generates some reference data for CCnorm
% you must run it under /tests/ref_stats.
rng(0, 'twister');

addpath(fullfile(pwd, 'CCnorm-master'));
n_neuron = 10;
n_trial = 50;
n_time = 100;
% first, generate data of std 100
y_data_all = randn(n_time,n_neuron);
y_data_all = reshape(y_data_all, [n_time, 1, n_neuron]);
% then, generate noise
y_data_noise = randn(n_time,n_trial,n_neuron);
y_data_all = bsxfun(@plus, y_data_all, y_data_noise);

cc_max_all = zeros(n_neuron,1);
% for loop to collect cc_max.
for i_neuron = 1:n_neuron
    y_data_this = y_data_all(:,:,i_neuron)';
    y_pred = mean(y_data_this, 1);
    % this is just dummy, not used by my ccmax function.
    y_pred = y_pred(:);
    [~, ~, cc_max_all(i_neuron)] = calc_CCnorm(y_data_this,y_pred);
end

% ok. save data.
h5create('ref_stats_ccnorm.hdf5', '/y_data_all', size(y_data_all));
h5write('ref_stats_ccnorm.hdf5', '/y_data_all', y_data_all);
h5create('ref_stats_ccnorm.hdf5', '/cc_max_all', size(cc_max_all));
h5write('ref_stats_ccnorm.hdf5', '/cc_max_all', cc_max_all);

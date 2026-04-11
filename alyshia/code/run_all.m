%% Fall Detection System — Master Pipeline
% Run from the submission/ directory.
%
% Pipeline:
%   step1 → load + filter raw data
%   step2 → extract 459 features per window
%   step3 → train RF with temporal+LOS features, LOPO CV
%   step4 → predict on test data, write CSVs

clear; clc; close all;
addpath('utils');

step1_load_filter;
step2_features;
step3_train;
step4_predict;

fprintf('========================================\n');
fprintf('  Pipeline complete.\n');
fprintf('  Predictions: submission/predictions/\n');
fprintf('========================================\n');

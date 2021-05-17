clearvars; clearvars -global; clc; close all;

addpath('Functions');
addpath(genpath('./Libraries'));

vl_setupnn();
InitParam();

[depthNet, colorNet] = LoadNetworks(true);

TrainSystem(depthNet, colorNet);
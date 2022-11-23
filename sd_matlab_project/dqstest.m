close all, clear all, clc;

SNR = -10;
print_num = 3;

figure;

for k=1:print_num
    file_path = sprintf("../RFML_dataset_new/2.PSK/QPSK_%ddB/QPSK_%ddB_%d.mat", SNR, SNR, k);
    load(file_path);
    t = 1:size(IQ, 2);
    subplot(2,3,k);
    scatter3(IQ(1, :), IQ(2, :), t, '*');
    axis([-5, 5, -5, 5, -inf, inf]);
end

for k=1:print_num
    file_path = sprintf("../RFML_dataset_new/0.Noise/Noise_%ddB/Noise_%ddB_%d.mat", SNR, SNR, k);
    load(file_path);
    t = 1:size(IQ, 2);
    subplot(2,3,k+3);
    scatter3(IQ(1, :), IQ(2, :), t, '*');
    axis([-5, 5, -5, 5, -inf, inf]);
end



function sequence = ar2sequence(ar,fs,L)
%ANALYSIS.AR2SEQUENCE converts arousals structures to arousal in bins.
%   sequence = ANALYSIS.AR2SEQUENCE(ar,fs,L) inputs an arousal structure "ar" and
%   iterates the structure to generate a arousal label vector in 1/fs bins.
%
%   Author: Andreas Brink-Kjaer.
%   Date: 17-Jun-2018
%
%   Input:  ar, arousal structure
%           fs, desired arousal label frequency
%           L, sequence length
%   Output: sequence, arousal vector with length L and frequency fs

if ~exist('fs','var')
    fs = 2;
end
if ~exist('L','var')
    L = 16*60*60*fs;
end
sequence = zeros(L,1);
for i = 1:ar.N
    idx = round(fs*(ar.start(i)-1)+1):round(fs*(ar.start(i)-1)+1 + fs*ar.duration(i) - 1);
    sequence(idx) = 1;
end
sequence = sequence(1:L);
end
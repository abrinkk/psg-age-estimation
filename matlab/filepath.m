function [ location ] = filepath( varargin )
%FILEPATH combines input strings in a directory format.
%   FILEPATH(varargin) joins input strings with appropriate use of "/" and
%   "\" to fit the correct system directory format.
%
%   Author: Caspar Aleksander Bang Jespersen
%
%   Input:  varargin, input stirngs
%   Output: location, joined strigns

if strcmp(varargin{end}(1),'.')
    args = varargin(1:end-1);
    location = fullfile(args{:});
    location = [location varargin{end}];
else
    location = fullfile(varargin{:});
end

end


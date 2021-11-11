%% SASREAD 
% *Read data from a SAS dataset*
%
%
%% Syntax
% |[numeric,txt,raw] = sasread(sasfile,xlsfile)|
%
% *Inputs*
%
% |sasfile| - full path to input SAS dataset 
%
% |xlsfile| - full path to Excel workbook containing |SAS2Excel| macro
%
% *Outputs*
%
% |numeric| - double array containing numeric data in |sasfile|
%
% |txt|     - cell string  array containing character data in |sasfile| 
%
% |raw|     - cell array containing  numeric and text data in |sasfile|
%
%
%% Description
%
% Employing ActiveX automation server, |sasread| creates an instance of Microsoft Excel, which, in its turn, creates 
% an instance of SAS and executes EXPORT procedure of SAS/ACCESS, transferring data from |sasfile| to an Excel worksheet
% - subsequently imported into Matlab with |xlsread|.
%
% Both SAS (including SAS/ACCESS Interface to PC Files) and Excel are needed to run |sasread|. Excel workbook containing |SAS2Excel| macro must be retained,
% its path provided in |xlsfile|.
% 
% |C:\sas.xls| is used for temporary data storage; if you do not have write access to |C:|, edit all references
% to |C:\sas.xls| in |sasread| and |SAS2Excel| macro. 
%
% |sasread| will ask you to close any open SAS sessions. 
%
% During transfer, you will be presented with a 'Save changes?' dialog by Excel; select 'Do not save changes'. (Suggestions on
% how to suppress this behavior are welcome).
%
%
%% Example
sasfile = 'C:\sasreaddemo.sas7bdat';         
xlsfile = 'C:\SAS-Matlab Converter.xls';

%%
dir('C:\*.sas7bdat')

%%
[numeric,text,raw] = sasread(sasfile,xlsfile)   %#ok

%%
class(numeric)
class(text)
class(raw)

%% See also
% |saswrite| (available from FEX)
% 
% |xlsread|
%
% |actxserver|
%
%
%Example: Oh, FEX code metrics..

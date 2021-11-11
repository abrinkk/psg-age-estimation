function[NUMERIC,TXT,RAW] = sasread(sasfile,xlsfile)
% SASREAD   Read data from a SAS dataset
% INPUTS    sasfile - full path to input SAS dataset 
%           xlsfile - full path to Excel workbook containing 'SAS2Excel' macro 
% OUTPUTS   numeric - double array containing numeric data in sasfile 
%           txt     - cell string  array containing character data in sasfile 
%           raw     - cell array containing  numeric and text data in sasfile
% NOTES     SASREAD creates an instance of Microsoft Excel, which opens a SAS 
%           instance  and  executes EXPORT procedure, transferring  data from
%           sasfile to an Excel worksheet - imported into Matlab with XLSREAD.
%           Both SAS and Excel are needed to run SASREAD. Excel workbook with
%           'SASToExcel' macro must be retained, its path provided in xlsfile.
%           (Note that Excel's involvement  limits the size of  datasets that 
%           can be handled). 'C:\sas.xls' is used for temporary data storage; 
%           if you do not have write access to disk C, edit all references to 
%           'C:\sas.xls' in this file and 'SAS2Excel' macro. SASREAD will ask
%           you to close any open SAS sessions. You will be  presented with a 
%           'Save changes?' dialog by Excel; select 'Do not save changes'.
% EXAMPLE   See SASREADDEMO
% SEE ALSO  SASWRITE (companion File Exchange submission), XLSREAD
% AUTHOR    Dimitri Shvorob, dimitri.shvorob@vanderbilt.edu, 11/1/05

f = 'C:\Users\andre\Dropbox\Phd\SleepAge\Scripts\matlab\sas.xls';
if nargin < 1
   error('Input argument ''sasfile'' is undefined')
end
if nargin < 2
   error('Input argument ''xlsfile'' is undefined')
end
if ~exist(xlsfile,'file')
   error('Read failed: could not find ''xlsfile''')
end   
if ~exist(sasfile,'file')
   error('Read failed: could not find ''sasfile''')
end   
try
   e = actxserver('Excel.Application');
catch
   error('Read failed: could not start Excel')
end    
try
   b = e.Workbooks.Open(xlsfile);
catch
   e.Quit
   error('Read failed: could not open ''xlsfile''')
end   
try
   s = b.Sheets.get('Item',1);
   c = s.get('Range','A1');
   m = findstr(sasfile,'.');
   if isempty(m) 
      c.Value = sasfile;
   else
      c.Value = sasfile(1:m-1);
   end   
catch
   e.Quit
   error('Read failed: ''xlsfile'' may have been corrupted')
end   
try
   e.ExecuteExcel4Macro('!SAS2Excel()');
catch
   e.Quit
   error('Read failed: macro in ''xlsfile'' encountered an error')
end
b.Close
e.Quit
[NUMERIC,TXT,RAW] = xlsread(f);
delete(f)
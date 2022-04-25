%%% convert the google-imagenet NEV files to CDTTables
%%% then convert those to a simpler array structure

% first setup the path
% couldn't get MATLABPATH env variable to work right
% so you might need to modify this if your toolbox is in a weird spot
import import_NEV.import_params_dir
import import_NEV.import_NEV_files

% for loading and saving
base_path = mfilename('fullpath'); % full path to this file
base_path = base_path(1:end-17); % get rid of the filename itself
base_nev_path = fullfile(base_path, '..', 'data', 'google-imagenet', 'smith-auto', 'raw');
base_cdt_path = fullfile(base_path, '..', 'data', 'google-imagenet', 'smith-auto', 'cdts');
base_array_path = fullfile(base_path, '..', 'data', 'google-imagenet', 'smith-auto', 'arrays');
param_path = fullfile(base_path, 'gaya_params.prototxt');
template_path = fullfile(base_path, 'gaya_template.prototxt');


% each filename
nev_list = dir(fullfile(base_nev_path, '*.nev'));
nev_list = {nev_list.name};
% full filepath for each one
full_nev_list = {};
for i = 1:length(nev_list)
    full_nev_list{i} = fullfile(base_nev_path, nev_list{i});
end

% this is mostly copied from the import_NEV demo
% but first we copy over the template files
copyfile(param_path, fullfile(import_params_dir(), 'gaya_params.prototxt'));
copyfile(template_path, fullfile(import_params_dir(), '..', 'trial_templates', 'gaya_template.prototxt'));
% these are needed to the start and end times of each trials, which are successful, etc.
proto_class = 'com.leelab.monkey_exp.ImportParamsProtos$ImportParams';
params_prototxt = fullfile(import_params_dir(), 'gaya_params.prototxt');
import_params = proto_functions.parse_proto_txt(params_prototxt, proto_class);

% now convert to CDT
% (should print number of successful trials for each one)
cdt_tables = import_NEV_files(full_nev_list, import_params);

% and save the results (in each own file)
for i = 1:length(nev_list)
    table = cdt_tables{i};
    save([base_cdt_path '/' nev_list{i}(1:end-3) 'mat'], 'table');
end

% now convert the CDTtables to arrays
% (also save which electrode-unit pairs the array contains)
for i = 1:length(cdt_tables)
    table = cdt_tables{i};
    [array, electrodes] = ArrayFromCDT(table, 1500); % second arg is time (1500 ms)
    save([base_array_path '/' nev_list{i}(1:end-3) 'mat'], 'array', 'electrodes');
end

%%% need to run this before running the processing files
%%% because importing things in matlab is tricky

% might need to manually change this if your toolbox is somewhere weird
% (couldn't get the MATLABPATH env variable to work)
relative_path = '../../yimeng_neural_analysis_toolbox';

addpath(genpath(relative_path));

% now run Yimeng's path initialization
% but that changes the working directory
% so keep track and move back
start_dir = pwd;
initialize_path;
initialize_java_path;
cd(start_dir);

% and everything should work from here

function [array, neurons] = ArrayFromCDT(cdt, time);
    % convert a CDT into a cell array of trial x neuron x time tensors
    % (the cell array allows variable numbers of trials per condition)
    % (because some days have slight data issues -- a few missing or extra trials)
    % also returns a list of the electrode-unit pairs corresponding to the neuron indices
    nargin
    
    if nargin == 1
        time = 1500; % reasonable default -- 500 ms + 500 on either side (?)
    end

    conditions = unique(cdt.condition);
    n_conditions = length(conditions);

    trials = histc(cdt.condition, conditions);

    neurons = AllNeurons(cdt);
    n_neurons = length(neurons);

    n_time = time; % just to reiterate

    array = {};

    % note that this doesn't work when a condition has no trials
    % (since that should never happen -- if it does might be a CTX file error) 
    for ith_im = 0:(n_conditions - 1)
        array{ith_im + 1} = zeros(trials(ith_im + 1), n_neurons, time, 'int8');

        cond_idxs = find(cdt.condition == ith_im);

        if length(cond_idxs) > 0
            for j = 1:trials(ith_im + 1)

                if j <= length(cond_idxs) % trying to deal with zero-spike trials
                    trial = cond_idxs(j);
                    
                    trial_spikes = cdt.spikeTimes{trial};
                    trial_active_elecs = cdt.spikeElectrode{trial};
                    trial_active_units = cdt.spikeUnit{trial};

                    trial_active_neurons = [trial_active_elecs trial_active_units];

                    active_neu_idxs = ismember(neurons, trial_active_neurons, 'rows');

                    for ith_neuron = 1:n_neurons

                        if active_neu_idxs(ith_neuron)
                            % then this neuron spiked this trial
                            trial_idx = sum(active_neu_idxs(1:ith_neuron));
                            spikes = trial_spikes{trial_idx} * 1000;
                            array{ith_im + 1}(j, ith_neuron, :) = histcounts(spikes, 0:time);

                        end
                    end
                end
            end
        end
    end
end

function neurons = AllNeurons(cdt)
    
    electrodes = vertcat(cdt.spikeElectrode{:});
    units = vertcat(cdt.spikeUnit{:});

    both = [electrodes units];

    neurons = unique(both, 'rows'); % all unique electrode-unit pairs
end

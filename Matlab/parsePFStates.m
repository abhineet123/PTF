states=importdata('../C++/MTF/log/states.txt');
states_diff=states(2:end, :)-repmat(states(1, :), size(states, 1)-1, 1);
states_diff_mean=mean(abs(states_diff), 1);
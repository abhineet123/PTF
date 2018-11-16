clear all;
% close all;

getParamLists;
colRGBDefs;
write_data = 0;
db_root_dir = '../../Datasets';
actor_ids = [2];
n_sub_seqs = 10;

% n_frames_list = [];
% for actor_id = actor_ids
%     if isempty(n_frames_list)
%         n_frames_list = actor_n_frames.data;
%     else
%         n_frames_list = vertcat(n_frames_list, actor_n_frames.data);
%     end
% end

n_actors = length(actor_ids);
sub_seq_n_frames = cell(n_actors, 1);
sub_seq_gaps = cell(n_actors, 1);
sub_seq_n_frames_list = cell(n_actors, 1);
n_frames = 0;
sub_seq_n_frames_sum = zeros(length(n_sub_seqs), 1);
cmbd_sub_seq_gaps = [];
cmbd_n_seq = 0;
n_sub_seq_id = 1;
for n_sub_seq = n_sub_seqs
    for actor_id = actor_ids
        actor = actors{actor_id + 1};
        actor_n_frames=importdata(sprintf('%s/%s/n_frames.txt', db_root_dir, actor));
        n_seq = length(actor_n_frames.data);
        sub_seq_start_ids = zeros(n_seq, n_sub_seq);        
        cmbd_n_seq = cmbd_n_seq + n_seq;
        sub_seq_n_frames{actor_id + 1} = zeros(n_seq, 1);
        sub_seq_gaps{actor_id + 1} = zeros(n_seq, 1);
        for seq_id = 1:n_seq            
            seq_n_frames = actor_n_frames.data(seq_id);
            n_frames = n_frames + seq_n_frames;
            sub_seq_gaps{actor_id + 1}(seq_id) = max(floor((seq_n_frames-1)/n_sub_seq), 1);
            start_id = 0;            
            for sub_seq_id = 1:n_sub_seq
                if start_id >= seq_n_frames
                    error('start_id: %d is too large for a sequence with %d frames', start_id, seq_n_frames);
                end
                sub_seq_n_frames{actor_id + 1}(seq_id) = sub_seq_n_frames{actor_id + 1}(seq_id) + seq_n_frames-start_id - 1;
                sub_seq_start_ids(seq_id, sub_seq_id) = start_id;
                start_id = start_id + sub_seq_gaps{actor_id + 1}(seq_id);
            end
            sub_seq_n_frames_sum(n_sub_seq_id) = sub_seq_n_frames_sum(n_sub_seq_id) + sub_seq_n_frames{actor_id + 1}(seq_id);
        end
        cmbd_sub_seq_gaps = [cmbd_sub_seq_gaps; sub_seq_gaps{actor_id + 1}];
        if write_data
            dlmwrite(sprintf('%s/%s/subseq_start_ids_%d.txt', db_root_dir, actor, n_sub_seqs),...
                sub_seq_start_ids, ',');
            dlmwrite(sprintf('%s/%s/subseq_n_frames_%d.txt', db_root_dir, actor, n_sub_seqs),...
                sub_seq_n_frames{actor_id + 1});
        end
    end
    n_sub_seq_id = n_sub_seq_id + 1;
end
n_frames = n_frames
sub_seq_n_frames_sum = sub_seq_n_frames_sum
increase_factor = sub_seq_n_frames_sum/n_frames
if length(n_sub_seqs)>1
    figure, plot(n_sub_seqs, sub_seq_n_frames);
    grid on;
    xlabel('n sub seqs');
    ylabel('sub seq n frames');
end
figure, plot(1:cmbd_n_seq, cmbd_sub_seq_gaps);
grid on;
xlabel('seq id');
ylabel('sub seq gap');

avg_sub_seq_gap = mean(cmbd_sub_seq_gaps)


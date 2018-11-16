if isnumeric(plot_data_desc{1}('value'))
    bars_per_group=length(plot_data_desc{1}('value'));
    labels=cell(n_lines, 1);
else
    bars_per_group = length(plot_data_desc{1}('line_style'));
    labels=cell(n_lines, 1);
end
colors=plot_data_desc{1}('color');
values=zeros(n_lines, bars_per_group);
stats_col = 1;
n_imported_fnames=0;
imported_fnames={};
imported_data={};
stats_file_names='';
key_strings='';
bar_locations = zeros(n_lines, bars_per_group);
for line_id=1:n_lines
    desc=plot_data_desc{line_id};
    curr_colors=desc('color');
    line_styles=desc('line_style');
    show_bar_legend_=show_bar_legend;
    if ~isempty(desc('stats_col'))
        show_bar_legend_=desc('stats_col');
        % stats_col=desc('stats_col');
        % fprintf('Getting data from mean of column %d\n', stats_col);
    end
    if ~isnumeric(desc('value'))
        if iscell(desc('stats_file_name'))
            key_strings = desc('value');
            key_string = key_strings{1};
        else
            key_string = desc('value');
        end        
        if ~isempty(desc('stats_file_name'))            
            if iscell(desc('stats_file_name'))
                stats_file_names = desc('stats_file_name');
                stats_file_name = stats_file_names{1};
            else
                stats_file_name = desc('stats_file_name');
            end
        else
            stats_file_name = 'tracking_stats.txt';
        end
        if isempty(stats_file_name) &&  isempty(key_string)
            values(line_id, 1) = mean(values(1:line_id-1, 1));
        else
            fprintf('Reading stats data for key %d: %s from %s\n',...
                line_id, key_string, stats_file_name);
            imported_fname_id=strmatch(stats_file_name, imported_fnames, 'exact');
            if ~isempty(imported_fname_id)
                stats=imported_data{imported_fname_id};
            else
                stats = importdata(stats_file_name);
                n_imported_fnames=n_imported_fnames+1;
                imported_fnames{n_imported_fnames}=stats_file_name;
                imported_data{n_imported_fnames}=stats;
            end        
            stats_col = 1;        
            %key_data=stats.data(~cellfun('isempty',strfind(stats.textdata(:, end),...
            %    key_string)), stats_col);
            key_data=stats.data(strncmp(stats.textdata(:, end),...
                key_string, length(key_string)), stats_col);

            key_data(isnan(key_data))=[];
            n_key_data=length(key_data);

            key_data_mean=mean(key_data);
            key_data_std=std(key_data);
            fprintf('key_data :: count: %d mean: %f std: %f\n',...
                n_key_data, key_data_mean, key_data_std);
            values(line_id, 1) = key_data_mean;
        end
        if bars_per_group==2
            if ~isempty(key_strings)
                key_string = key_strings{2};
            end        
            if ~isempty(stats_file_names)
                stats_file_name = stats_file_names{2};
                if isempty(stats_file_name) &&  isempty(key_string)
                    values(line_id, 2) = mean(values(1:line_id-1, 2));
                else
                    fprintf('Reading second stats data for key %d: %s from %s\n',...
                            line_id, key_string, stats_file_name);
                    imported_fname_id=strmatch(stats_file_name, imported_fnames, 'exact');
                    if ~isempty(imported_fname_id)
                        stats=imported_data{imported_fname_id};
                    else
                        stats = importdata(stats_file_name);
                        n_imported_fnames=n_imported_fnames+1;
                        imported_fnames{n_imported_fnames}=stats_file_name;
                        imported_data{n_imported_fnames}=stats;
                    end    
                    % stats = importdata(stats_file_name);
                    key_data=stats.data(strncmp(stats.textdata(:, end),...
                        key_string, length(key_string)), stats_col);

                    key_data(isnan(key_data))=[];
                    n_key_data=length(key_data);

                    key_data_mean=mean(key_data);
                    key_data_std=std(key_data);
                    fprintf('key_data :: count: %d mean: %f std: %f\n',...
                        n_key_data, key_data_mean, key_data_std);
                    values(line_id, 2) = key_data_mean;
                end
            else
                if isempty(stats_file_name) &&  isempty(key_string)
                    values(line_id, 2) = mean(values(1:line_id-1, 2));
                else
                    values(line_id, 2) = key_data_std;
                end
            end
        end
        %         labels{2*line_id-1}=sprintf('%s/Mean', desc('label'));
        %         labels{2*line_id}=sprintf('%s/Std', desc('label'));        
        labels{line_id}=desc('label');    
    else
        labels{line_id}=desc('label');
        values(line_id, :)=desc('value');
    end
    % bar_x=[2*line_id-1, 2*line_id];
    line_id_avg=(line_id-1)*inter_bar_gap_ratio + 0.40;
    bar_locations(line_id, 1) = line_id_avg;
    if ~horz_bar_plot
        bar_ids(line_id) = bar(line_id_avg, values(line_id, 1),...
            'Parent', gca,...
            'BarWidth', bar_width,...
            'LineWidth', bar_line_width,...
            'LineStyle', line_styles{1},...
            'FaceColor', col_rgb{strcmp(col_names,curr_colors(1))},...
            'EdgeColor', col_rgb{strcmp(col_names,'black')});
    else
        bar_ids(line_id) = barh(line_id_avg, values(line_id, 1),...
            'Parent', gca,...
            'BarWidth', bar_width,...
            'LineWidth', bar_line_width,...
            'LineStyle', line_styles{1},...
            'FaceColor', col_rgb{strcmp(col_names,curr_colors(1))},...
            'EdgeColor', col_rgb{strcmp(col_names,'black')});
    end
    if length(line_styles)>1
        bar_locations(line_id, 2) = line_id_avg + intra_bar_gap;
        if ~horz_bar_plot
            bar(line_id_avg + intra_bar_gap, values(line_id, 2),...
                'Parent', gca,...
                'BarWidth', bar_width,...
                'LineWidth', bar_line_width,...
                'LineStyle', line_styles{2},...
                'FaceColor', col_rgb{strcmp(col_names,curr_colors(1))},...
                'EdgeColor', col_rgb{strcmp(col_names,'black')});
        else
            barh(line_id_avg + intra_bar_gap, values(line_id, 2),...
                'Parent', gca,...
                'BarWidth', bar_width,...
                'LineWidth', bar_line_width,...
                'LineStyle', line_styles{2},...
                'FaceColor', col_rgb{strcmp(col_names,curr_colors(1))},...
                'EdgeColor', col_rgb{strcmp(col_names,'black')});
        end                
    end
    if bar_with_legend
        if annotate_bar_legend
            if annotate_with_ratio
                labels{line_id}=sprintf('%s:%.2f',desc('label'),...
                    values(line_id, 1)/values(line_id, 2));
            else
                labels{line_id}=sprintf('%s:%.2f',desc('label'),...
                    values(line_id, 1));
            end
        end
    end
    if annotate_bars
        for bar_id=1:bars_per_group
            if isempty(annotation_col)
                bar_annotation_col=col_rgb{strcmp(col_names,curr_colors{bar_id})};
            else
                bar_annotation_col = annotation_col;
            end
            annotation('textbox',...
                [0 0 0.3 0.15],...
                'String', sprintf('%.2f',values(line_id, bar_id)),...
                'FontSize',annotation_font_size,...
                'FontWeight','bold',...
                'FontName','Times New Roman',...
                'LineStyle','-',...
                'EdgeColor','none',...
                'LineWidth',2,...
                'BackgroundColor','none',...
                'Color',bar_annotation_col,...
                'FitBoxToText','on');
        end
    end
end
if bar_with_legend
    if show_bar_legend_
        if col_legend
            h_legend=columnlegend(2,labels,'NorthWest', 'boxon');
        else
            h_legend=legend(bar_ids, labels);
        end
        set(h_legend,'FontSize',legend_font_size);
        set(h_legend,'FontWeight','bold');
    end
    set(gca, 'XAxisLocation', 'bottom');
    set(gca, 'YAxisLocation', 'left');
    set(gca, 'Color', 'None');
    set(gca,'box','off');
    if ~horz_bar_plot        
        x_lim=(n_lines-1)*inter_bar_gap_ratio + 1.40;
        set(gca, 'XLim', [0, x_lim]);
        %     set(gca, 'XTick', 1:n_lines);
        set(gca, 'XTickLabel', []);        
        ylabel('Speed in FPS');
    else
        y_lim=(n_lines-1)*inter_bar_gap_ratio + 1.40;
        set(gca, 'YLim', [0, y_lim]);
        %     set(gca, 'XTick', 1:n_lines);
        set(gca, 'YTickLabel', []);
        xlabel('Speed in FPS');
    end
else
    if horz_bar_plot
        % b = barh(values);
        set(gca, 'YTick', bar_locations(:, 1));
        set(gca, 'YTickLabel', labels, 'DefaultTextInterpreter', 'none');
        xlabel('MCD Error');
    else
        % b=bar(values);
        set(gca, 'XTick', bar_locations(:, 1));
        ylabel('MCD Error');
        set(gca, 'XTickLabel', labels, 'DefaultTextInterpreter', 'none');
    end
%     line_styles=plot_data_desc{1}('line_style');
%     for bar_id=1:bars_per_group
%         set(b(bar_id), 'LineStyle', line_styles{bar_id});
%         set(b(bar_id), 'FaceColor', col_rgb{strcmp(col_names,colors{bar_id})});
%         set(b(bar_id), 'EdgeColor', col_rgb{strcmp(col_names,'black')});
%     end
end
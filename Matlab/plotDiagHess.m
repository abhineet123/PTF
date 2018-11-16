Std_data=plot_data{strcmp(data_types,'Std')};
if plot_sec_ord_hess
    Std2_data=plot_data{strcmp(data_types,'Std2')};
end
if plot_misc_hess
    InitSelf_data=plot_data{strcmp(data_types,'InitSelf')};
    CurrSelf_data=plot_data{strcmp(data_types,'CurrSelf')};
    SumOfStd_data=plot_data{strcmp(data_types,'SumOfStd')};
    SumOfSelf_data=plot_data{strcmp(data_types,'SumOfSelf')};
    if plot_sec_ord_hess
        InitSelf2_data=plot_data{strcmp(data_types,'InitSelf2')};
        CurrSelf2_data=plot_data{strcmp(data_types,'CurrSelf2')};
        SumOfStd2_data=plot_data{strcmp(data_types,'SumOfStd2')};
        SumOfSelf2_data=plot_data{strcmp(data_types,'SumOfSelf2')};
    end
end
if plot_num_hess
    Hessian_data=plot_data{strcmp(data_types,'Hessian')};
end
plot_id=1;
for state_id = state_ids
    diag_fig=diag_figs{state_id};
    set(0,'CurrentFigure',diag_fig);
    %                 figure(diag_fig);
    x_label = sprintf('param %d frame %d', state_id, frame_id);
    set(diag_fig, 'Name', x_label);
    plot_title=sprintf('Different Hessians for %s (%s frame %d)',...
        upper(am_name), seq_name, frame_id );
    title(plot_title, 'interpreter','none');
    plot_legend={};
    if plot_sec_ord_hess
        if frame_id==start_id
            hold off;
            plot_handles{plot_id}=plot(Std2_data(:, 2*state_id-1), Std2_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'red')}, 'LineWidth', line_width);
            plot_legend=[plot_legend, {'Std2'}];
        else
            set(plot_handles{plot_id},'ydata',Std2_data(:, 2*state_id));
            %             set(axes_handles{axis_id},'title',plot_title);
        end
        plot_id=plot_id+1;
        
        if opt_type==0
            hac_val=min(Std2_data(:, 2*state_id));
        else
            hac_val=max(Std2_data(:, 2*state_id));
        end
        hac_data(1:diag_res)=hac_val;
        hold on;
        if frame_id==start_id
            plot_handles{plot_id}=plot(Std2_data(:, 2*state_id-1), hac_data,...
                'Color', col_rgb{strcmp(col_names,'red')},...
                'LineWidth', line_width, 'LineStyle', '--');
            plot_legend=[plot_legend, {'HAC'}];
        else
            set(plot_handles{plot_id},'ydata',hac_data);
        end
        plot_id=plot_id+1;
    end    
    if plot_misc_hess
        if frame_id==start_id
            plot_handles{plot_id}=plot(InitSelf_data(:, 2*state_id-1), InitSelf_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'cyan')}, 'LineWidth', line_width);
            plot_legend=[plot_legend, {'InitSelf'}];
        else
            set(plot_handles{plot_id},'ydata',InitSelf_data(:, 2*state_id));
        end
        plot_id=plot_id+1;
        
        hold on;
        if frame_id==start_id
            plot_handles{plot_id}=plot(CurrSelf_data(:, 2*state_id-1), CurrSelf_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'blue')}, 'LineWidth', line_width);
            plot_legend=[plot_legend, {'CurrSelf'}];
        else
            set(plot_handles{plot_id},'ydata',CurrSelf_data(:, 2*state_id));
        end
        plot_id=plot_id+1;
        
        if frame_id==start_id
            plot_handles{plot_id}=plot(SumOfSelf_data(:, 2*state_id-1), SumOfSelf_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'black')}, 'LineWidth', line_width);%
            plot_legend=[plot_legend, {'SumOfSelf'}];
        else
            set(plot_handles{plot_id},'ydata',SumOfSelf_data(:, 2*state_id));
        end
        plot_id=plot_id+1;
        
        
        if frame_id==start_id
            plot_handles{plot_id}=plot(SumOfStd_data(:, 2*state_id-1), SumOfStd_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'orange')}, 'LineWidth', line_width);
            plot_legend=[plot_legend, {'SumOfStd'}];
        else
            set(plot_handles{plot_id},'ydata',SumOfStd_data(:, 2*state_id));
        end
        plot_id=plot_id+1;
        
        
        if plot_sec_ord_hess
            
            if frame_id==start_id
                plot_handles{plot_id}=plot(InitSelf2_data(:, 2*state_id-1), InitSelf2_data(:, 2*state_id),...
                    'Color', col_rgb{strcmp(col_names,'cyan')}, 'LineWidth', line_width);
                plot_legend=[plot_legend, {'InitSelf2'}];
            else
                set(plot_handles{plot_id},'ydata',InitSelf2_data(:, 2*state_id));
            end
            plot_id=plot_id+1;
            
            if frame_id==start_id
                plot_handles{plot_id}=plot(CurrSelf2_data(:, 2*state_id-1), CurrSelf2_data(:, 2*state_id),...
                    'Color', col_rgb{strcmp(col_names,'blue')}, 'LineWidth', line_width);
                plot_legend=[plot_legend, {'CurrSelf2'}];
            else
                set(plot_handles{plot_id},'ydata',CurrSelf2_data(:, 2*state_id));
            end
            plot_id=plot_id+1;
            
            if frame_id==start_id
                plot_handles{plot_id}=plot(SumOfSelf2_data(:, 2*state_id-1), SumOfSelf2_data(:, 2*state_id),...
                    'Color', col_rgb{strcmp(col_names,'black')}, 'LineWidth', line_width);%
                plot_legend=[plot_legend, {'SumOfSelf2'}];
            else
                set(plot_handles{plot_id},'ydata',SumOfSelf2_data(:, 2*state_id));
            end
            plot_id=plot_id+1;
            
            
            if frame_id==start_id
                plot_handles{plot_id}=plot(SumOfStd2_data(:, 2*state_id-1), SumOfStd2_data(:, 2*state_id),...
                    'Color', col_rgb{strcmp(col_names,'orange')}, 'LineWidth', line_width);
                plot_legend=[plot_legend, {'SumOfStd2'}];
            else
                set(plot_handles{plot_id},'ydata',SumOfStd2_data(:, 2*state_id));
            end
            plot_id=plot_id+1;
        end
        
    end
    if plot_num_hess
        if frame_id==start_id
            plot_handles{plot_id}=plot(Hessian_data(:, 2*state_id-1), Hessian_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'green')});
            plot_legend=[plot_legend, {'Numerical'}];
        else
            set(plot_handles{plot_id},'ydata',Hessian_data(:, 2*state_id));
        end
        plot_id=plot_id+1;
    end
    
    
    hold on;
    if frame_id==start_id
        plot_handles{plot_id}=plot(Std_data(:, 2*state_id-1), Std_data(:, 2*state_id),...
            'Color', col_rgb{strcmp(col_names,'magenta')}, 'LineWidth', line_width);
        plot_legend=[plot_legend, {'Std'}];
        % xlabel(['$T_x$'], 'interpreter','latex');
        xlabel([sprintf('$S_{%d}$', state_id)], 'interpreter','latex');
        ylabel([sprintf('$\\frac{\\partial^2 F_{%s}}{\\partial S_{%d}^2}$', am_name, state_id)], 'interpreter','latex');
        xlhand = get(gca,'xlabel');
        ylhand = get(gca,'ylabel');
        set(xlhand,'fontsize',50);
        set(ylhand,'fontsize',50);
        legend(plot_legend);
        grid on;
    else
        set(plot_handles{plot_id},'ydata',Std_data(:, 2*state_id));
    end
    plot_id=plot_id+1;
    
end
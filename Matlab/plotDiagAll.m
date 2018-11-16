if plot_feat_norm
    Norm_data=plot_data{strcmp(data_types,'FeatNorm')};
    if invert_feat_norm
        for state_id = state_ids
            Norm_data(:, 2*state_id) = -Norm_data(:, 2*state_id);
        end
    end
elseif plot_likelihood
    Norm_data=plot_data{strcmp(data_types,'Likelihood')};    
else
    Norm_data=plot_data{strcmp(data_types,'Norm')};
end

if ~plot_only_norm || plot_num_jac
    Jacobian_data=plot_data{strcmp(data_types,'Jacobian')};
end
if ~plot_only_norm
    StdJac_data=plot_data{strcmp(data_types,'StdJac')};
    Std_data=plot_data{strcmp(data_types,'Std')};
    Std2_data=plot_data{strcmp(data_types,'Std2')};
    
    Hessian_data=plot_data{strcmp(data_types,'Hessian')};
    if plot_misc_hess
        InitSelf_data=plot_data{strcmp(data_types,'InitSelf')};
        CurrSelf_data=plot_data{strcmp(data_types,'CurrSelf')};
        SumOfStd_data=plot_data{strcmp(data_types,'SumOfStd')};
        SumOfSelf_data=plot_data{strcmp(data_types,'SumOfSelf')};
    end
end
plot_id=1;
axis_id=1;

if plot_only_norm    
    plot_rows = 1;
    if plot_num_jac
        plot_cols = 2;
    else
        plot_cols = 1;
    end    
else
    plot_rows = 2;
    plot_cols = 2;
end

for state_id = state_ids
    if plot_only_norm && plot_norm_in_one_fig && state_id>1
        close(diag_figs{state_id});
    else
        diag_fig=diag_figs{state_id};
        %         set (diag_fig, 'Units', 'Normalized', 'Position', [0,0,1,1]);
        x_label = sprintf('param %d frame %d', state_id, frame_id);
        set(diag_fig, 'Name', x_label);
    end   
    
    % matlab_StdJac=gradient(Norm_data(:, 2*state_id));
    % matlab_Std=gradient(StdJac_data(:, 2*state_id));  
    
    if plot_feat_norm
        y_label=sprintf('D_f');
        plot_title=sprintf('%s FeatNorm', upper(am_name_disp));
    elseif plot_likelihood
        y_label=sprintf('L_f');
        plot_title=sprintf('%s Likelihood', upper(am_name_disp));
    else
        plot_title=sprintf('%s %s frame %d', upper(am_name_disp), seq_name, frame_id);
        y_label=sprintf('f_{%s}', lower(am_name_disp));
        if plot_num_likelihood && isempty(diag_lkl_figs{state_id})
            diag_lkl_figs{state_id}=figure;
        end
    end
    if ~isempty(diag_lkl_figs{state_id})      
        Norm_likelihood_data = getLikelihood(Norm_data(:, 2*state_id),...
            likelihood_alpha,likelihood_beta,likelihood_type);
    end
    
    if plot_norm_in_one_fig
        if frame_id==start_id
            figure(diag_figs{1});
            if state_id==1
                hold off;
            else
                hold on;
            end 
            plot_handles{plot_id}=plot(Norm_data(:, 2*state_id-1), Norm_data(:, 2*state_id),...
                'LineWidth', line_width,...
                'Color', col_rgb{strcmp(col_names,plot_cols_123(state_id))});
            grid on;
            x_label = sprintf('t_x / t_y');
            if state_id==state_id(end)
                title(plot_title, 'interpreter', 'none'), xlabel(x_label), ylabel(y_label), legend(plot_legend_123);   
            end
        else
            set(plot_handles{plot_id},'ydata',Norm_data(:, 2*state_id));
            figure(diag_figs{1});
            title(plot_title, 'interpreter', 'none')
        end
    else
        if frame_id==start_id
            if ~isempty(diag_lkl_figs{state_id})
                figure(diag_lkl_figs{state_id}), hold off;
                lkl_plot_title=sprintf('%s Norm Likelihood %.2fa%.2fb',...
                    upper(am_name), likelihood_alpha, likelihood_beta);  
                lkl_plot_handles{plot_id} = plot(Norm_data(:, 2*state_id-1), Norm_likelihood_data,...
                'LineWidth', line_width);
                title(lkl_plot_title, 'Interpreter', 'none'), grid on, xlabel(x_label), ylabel(lkl_plot_title, 'Interpreter', 'none');
            end
            figure(diag_fig), hold off;
            axes_handles{axis_id}=subplot(plot_rows,plot_cols,1);
            plot_handles{plot_id}=plot(Norm_data(:, 2*state_id-1), Norm_data(:, 2*state_id),...
                'LineWidth', line_width);
            title(plot_title), grid on, xlabel(x_label), ylabel(plot_title);
        else
            set(plot_handles{plot_id},'ydata',Norm_data(:, 2*state_id));
            %             set(axes_handles{axis_id},'title',plot_title);
            if ~isempty(lkl_plot_handles{plot_id})
                set(lkl_plot_handles{plot_id},'ydata',Norm_likelihood_data);
            end
        end
    end

    axis_id=axis_id+1;
    plot_id=plot_id+1;

    if ~plot_only_norm || plot_num_jac
        plot_title=sprintf('%s Jacobian', upper(am_name));
        if frame_id==start_id
            axes_handles{axis_id}=subplot(plot_rows,plot_cols,2);
            hold off;
            plot_handles{plot_id}=plot(Jacobian_data(:, 2*state_id-1), Jacobian_data(:, 2*state_id),...
                'LineWidth', line_width, 'Color', col_rgb{strcmp(col_names,'green')});
           grid on, xlabel(x_label), ylabel('Jacobian');
        else
            set(plot_handles{plot_id},'ydata',Jacobian_data(:, 2*state_id));
        end
        axis_id=axis_id+1;
        plot_id=plot_id+1;
    end
    
    if plot_only_norm
       if plot_num_jac
        title(sprintf('%s Numerical Jacobian', upper(am_name)));
       end       
       continue;
    end       
    
    if frame_id==start_id
        hold on;
        plot_handles{plot_id}=plot(StdJac_data(:, 2*state_id-1), StdJac_data(:, 2*state_id),...
            'Color', col_rgb{strcmp(col_names,'red')});
         title('Jacobian'), legend('numerical', 'analytical');
    else
        set(plot_handles{plot_id},'ydata',StdJac_data(:, 2*state_id));
        %             set(axes_handles{axis_id},'title',plot_title);
    end
    
    plot_id=plot_id+1;  
    

    %  plot(Jacobian_data(:, 2*state_id-1), matlab_StdJac,...
    %'Color', col_rgb{strcmp(col_names,'blue')});
    
    
    plot_title=sprintf('%s Std', upper(am_name));
    if frame_id==start_id
        axes_handles{axis_id}=subplot(plot_rows,plot_cols,3);
        hold off;
        plot_handles{plot_id}=plot(Std_data(:, 2*state_id-1), Std_data(:, 2*state_id),...
            'LineWidth', line_width, 'Color', col_rgb{strcmp(col_names,'red')});
    else
        set(plot_handles{plot_id},'ydata',Std_data(:, 2*state_id));
        %             set(axes_handles{axis_id},'title',plot_title);
    end
    axis_id=axis_id+1;
    plot_id=plot_id+1;
    
    hold on;
    if frame_id==start_id
        plot_handles{plot_id}=plot(Hessian_data(:, 2*state_id-1), Hessian_data(:, 2*state_id),...
            'LineWidth', line_width, 'Color', col_rgb{strcmp(col_names,'green')});
        title(plot_title), grid on, xlabel(x_label), ylabel('Std'), legend('analytical', 'numerical');
    else
        set(plot_handles{plot_id},'ydata',Hessian_data(:, 2*state_id));
    end
    plot_id=plot_id+1;
    
    
    plot_title=sprintf('Std2');
    if frame_id==start_id
        axes_handles{axis_id}=subplot(plot_rows,plot_cols,4);
        hold off;
        plot_handles{plot_id}=plot(Std2_data(:, 2*state_id-1), Std2_data(:, 2*state_id),...
            'Color', col_rgb{strcmp(col_names,'red')});
        plot_legend={'Std2'};
    else
        set(plot_handles{plot_id},'ydata',Std2_data(:, 2*state_id));
        %             set(axes_handles{axis_id},'title',plot_title);
    end
    axis_id=axis_id+1;
    plot_id=plot_id+1;
    
    hold on;
    if plot_misc_hess
        if frame_id==start_id
            plot_handles{plot_id}=plot(InitSelf_data(:, 2*state_id-1), InitSelf_data(:, 2*state_id),...
                'LineWidth', line_width, 'Color', col_rgb{strcmp(col_names,'cyan')});
            plot_legend=[plot_legend, {'InitSelf'}];
        else
            set(plot_handles{plot_id},'ydata',InitSelf_data(:, 2*state_id));
        end
        plot_id=plot_id+1;
        
        if frame_id==start_id
            plot_handles{plot_id}=plot(CurrSelf_data(:, 2*state_id-1), CurrSelf_data(:, 2*state_id),...
                'LineWidth', line_width, 'Color', col_rgb{strcmp(col_names,'blue')});
            plot_legend=[plot_legend, {'CurrSelf'}];
        else
            set(plot_handles{plot_id},'ydata',CurrSelf_data(:, 2*state_id));
        end
        plot_id=plot_id+1;
        
        if frame_id==start_id
            plot_handles{plot_id}=plot(SumOfStd_data(:, 2*state_id-1), SumOfStd_data(:, 2*state_id),...
                'LineWidth', line_width, 'Color', col_rgb{strcmp(col_names,'orange')});
            plot_legend=[plot_legend, {'SumOfStd'}];
        else
            set(plot_handles{plot_id},'ydata',SumOfStd_data(:, 2*state_id));
        end
        plot_id=plot_id+1;
        
        if frame_id==start_id
            plot_handles{plot_id}=plot(SumOfSelf_data(:, 2*state_id-1), SumOfSelf_data(:, 2*state_id),...
                'LineWidth', line_width, 'Color', col_rgb{strcmp(col_names,'black')});
            plot_legend=[plot_legend, {'SumOfSelf'}];
        else
            set(plot_handles{plot_id},'ydata',SumOfSelf_data(:, 2*state_id));
        end
        plot_id=plot_id+1;
    end
    
    if frame_id==start_id
        plot_handles{plot_id}=plot(Hessian_data(:, 2*state_id-1), Hessian_data(:, 2*state_id),...
            'LineWidth', line_width, 'Color', col_rgb{strcmp(col_names,'green')});
        plot_legend=[plot_legend, {'numerical'}];
        title(plot_title), grid on, xlabel(x_label), ylabel('Std2'), legend(plot_legend);
    else
        set(plot_handles{plot_id},'ydata',Hessian_data(:, 2*state_id));
    end
    plot_id=plot_id+1;
end
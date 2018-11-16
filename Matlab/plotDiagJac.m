Norm_data=plot_data{strcmp(data_types,'Norm')};
StdJac_data=plot_data{strcmp(data_types,'StdJac')};
if(plot_num_jac)
    Jacobian_data=plot_data{strcmp(data_types,'Jacobian')};
end
plot_id=1;
for state_id = state_ids
    diag_fig=diag_figs{state_id};
    set(0,'CurrentFigure',diag_fig);
    plot_title=sprintf('Norm');
    fig_title=sprintf('Norm and StdJac for %s (%s frame %d)',...
        upper(am_name), seq_name, frame_id );
    set(diag_fig, 'Name', fig_title);
    if frame_id==start_id
        subplot(1,2,1);
        plot_handles{plot_id}=plot(Norm_data(:, 2*state_id-1), Norm_data(:, 2*state_id),...
            'LineWidth', line_width);
        title(plot_title, 'interpreter','none'), grid on;
        % xlabel(['$T_x$'], 'interpreter','latex');
        xlabel([sprintf('$S_{%d}$', state_id)], 'interpreter','latex');
        ylabel([sprintf('$F_{%s}$', am_name)], 'interpreter','latex');
        xlhand = get(gca,'xlabel');
        ylhand = get(gca,'ylabel');
        set(xlhand,'fontsize',50);
        set(ylhand,'fontsize',50);
    else
        set(plot_handles{plot_id},'ydata',Norm_data(:, 2*state_id));
    end
    
    hold on;
    plot_id=plot_id+1;
    plot_title=sprintf('StdJac');
    if frame_id==start_id
        subplot(1,2,2);
        plot_handles{plot_id}=plot(StdJac_data(:, 2*state_id-1),...
            StdJac_data(:, 2*state_id), 'LineWidth', line_width,...
            'Color', col_rgb{strcmp(col_names,'red')});
        if(plot_num_jac)
            hold on;
            plot_id=plot_id+1;
            plot_handles{plot_id}=plot(Jacobian_data(:, 2*state_id-1),...
                Jacobian_data(:, 2*state_id), 'LineWidth', line_width,...
                'Color', col_rgb{strcmp(col_names,'green')});
            legend('analytical', 'numerical');
        end
        title(plot_title, 'interpreter','none'), grid on;
        xlabel(['$T_x$'], 'interpreter','latex');
        xlabel([sprintf('$S_{%d}$', state_id)], 'interpreter','latex');
        ylabel([sprintf('$\\frac{\\partial F_{%s}}{\\partial S_{%d}}$', am_name, state_id)], 'interpreter','latex');
        xlhand = get(gca,'xlabel');
        ylhand = get(gca,'ylabel');
        set(xlhand,'fontsize',50);
        set(ylhand,'fontsize',50);
    else
        set(plot_handles{plot_id},'ydata',StdJac_data(:, 2*state_id));
        if(plot_num_jac)
            hold on;
            plot_id=plot_id+1;
            set(plot_handles{plot_id},'ydata',Jacobian_data(:, 2*state_id));
        end
    end
    plot_id=plot_id+1;
end
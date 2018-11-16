function stacked_img = stackImages(img_list, stack_order)
    n_images = length(img_list)
    img_size = size(img_list{1})
    grid_size = ceil(sqrt(n_images));
    stacked_img = None;
    list_ended = False;
    inner_axis = 1 - stack_order;
    for row_id = 1:grid_size
        start_id = grid_size * row_id;
        curr_row = '';
        for col_id = 1:grid_size
            img_id = start_id + col_id;
            if img_id >= n_images
                curr_img = zeros(img_size)
                list_ended = True;
            else
                curr_img = img_list{img_id};
                if img_id == n_images - 1
                    list_ended = True;
                end
            end
            if isempty(curr_row)
                curr_row = curr_img
            else
                curr_row = cat(1, curr_row, curr_img);
            end
        end
        if isempty(stacked_img)
            stacked_img = curr_row
        else
            stacked_img = cat(stack_order, stacked_img, curr_row);
        end
        if list_ended
            break;
        end
    end
    return stacked_img;
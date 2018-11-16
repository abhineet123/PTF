function rectangle_t = fit_rectangle(boundary_points)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author - Onkar Raut, University of North Carolina at Charlotte
% Based on - 
% fit_rectangle -   Function provides a least squares fit to the 
%                   given boundary points of an object of unknown shape. 
% 
% Inputs - Boundary elements that must be a Nx2 array. (atleast 3 required)
% Output -
%       Bounding_points(4x2):
%       Function will return a struct consisting of the bounding points
%       equation_of_diagonals(2x2):
%       Function will return the equation of the diagonal in the form
%       y = mx +c (giving back m and c foreach diagonal). 
%       equation_of_bounding_sides(4x2):
%       Equation of the Bounding sides in the form y = mx +c
%       centroid(1x2): 
%       Location of the centroid in cartesian coordinates
m = boundary_points(:,1);
n = boundary_points(:,2);

centroid = [sum(m)/size(m,1) sum(n)/size(n,1)];
diff = [m n] - (centroid'*ones(1,size(m,1)))';
theta_maj = 0.5*atan(2*sum(diff(:,1).*diff(:,2))/sum(diff(:,1).^2-diff(:,2).^2));
theta_min = theta_maj(1,1)+pi;

class_ud = diff(:,2) - tan(theta_maj(1,1)).*diff(:,1);
class_lr = diff(:,2) - cot(theta_maj(1,1)).*diff(:,1);

max_up = find(class_ud == max(class_ud));
max_down = find(class_ud == min(class_ud));
extremes_ud = [m(max_up,1) n(max_up,1); m(max_down,1) n(max_down,1)];

max_l = find(class_lr == max(class_lr));
max_r = find(class_lr == min(class_lr));
extremes_lr= [m(max_l,1) n(max_l,1); m(max_r,1) n(max_r,1)];

x_1 = extremes_ud(1,1); y_1 = extremes_ud(1,2);
x_2 = extremes_ud(2,1); y_2 = extremes_ud(2,2);
x_3 = extremes_lr(1,1); y_3 = extremes_lr(1,2);
x_4 = extremes_lr(2,1); y_4 = extremes_lr(2,2);

t_lx(1,1) = (x_1*tan(theta_maj(1,1))+x_3*cot(theta_maj(1,1))+y_3-y_1)/(tan(theta_maj(1,1))+cot(theta_maj(1,1)));
t_ly(1,1) = (y_1*cot(theta_maj(1,1))+y_3*tan(theta_maj(1,1))+x_3-x_1)/(tan(theta_maj(1,1))+cot(theta_maj(1,1)));
t_rx(1,1) = (x_1*tan(theta_maj(1,1))+x_4*cot(theta_maj(1,1))+y_4-y_1)/(tan(theta_maj(1,1))+cot(theta_maj(1,1)));
t_ry(1,1) = (y_1*cot(theta_maj(1,1))+y_4*tan(theta_maj(1,1))+x_4-x_1)/(tan(theta_maj(1,1))+cot(theta_maj(1,1)));
    
b_lx(1,1) = (x_2*tan(theta_maj(1,1))+x_3*cot(theta_maj(1,1))+y_3-y_2)/(tan(theta_maj(1,1))+cot(theta_maj(1,1)));
b_ly(1,1) = (y_2*cot(theta_maj(1,1))+y_3*tan(theta_maj(1,1))+x_3-x_2)/(tan(theta_maj(1,1))+cot(theta_maj(1,1)));
b_rx(1,1) = (x_2*tan(theta_maj(1,1))+x_4*cot(theta_maj(1,1))+y_4-y_2)/(tan(theta_maj(1,1))+cot(theta_maj(1,1)));
b_ry(1,1) = (y_2*cot(theta_maj(1,1))+y_4*tan(theta_maj(1,1))+x_4-x_2)/(tan(theta_maj(1,1))+cot(theta_maj(1,1)));

bounding_points = [t_lx,t_ly;
        t_rx,t_ry;
        b_lx,b_ly;
        b_rx,b_ry];
m_1 = (t_ly-b_ry)/(t_lx-b_rx);
m_2 = (t_ry-b_ly)/(t_rx-b_lx);
equation_of_diagonals = [ m_1 -m_1*b_rx+b_ry;
                          m_2 -m_2*b_lx+b_ly     ];
m_1 = (t_ly-t_ry)/(t_lx-t_rx);
m_2 = (b_ly-b_ry)/(b_lx-b_rx);

m_3 = (t_ry-b_ry)/(t_rx-b_rx);
m_4 = (t_ly-b_ly)/(t_lx-b_lx);
equation_of_bounding_sides = [ m_1 -m_1*t_rx+t_ry
                               m_2 -m_2*b_rx+b_ry
                               m_3 -m_3*b_rx+b_ry
                               m_4 -m_4*b_lx+b_ly ];
rectangle_t = struct( ...
    'bounding_points', bounding_points, ...
    'equation_of_diagonals',equation_of_diagonals, ...
    'equation_of_bounding_sides',equation_of_bounding_sides, ...
    'centroid',centroid);
end
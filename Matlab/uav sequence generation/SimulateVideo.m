% get trajectory

close all
clear 

x_gt = [1340, 534;
        1521, 512;
        1543, 702;
        1372, 712];

%% ground truth%%
X_gt = [1444, 612;
        1402, 734;
        1343, 867;
        1266, 988;
        1199, 1072;
        1157, 1193;
        943, 1120;
        921, 1008;
        1011, 902;
        1024, 789;
        1156, 696;
        1444, 612];
I2 = imread('YJ_map_right.bmp');
figure(1);
imshow(I2);
hold on
plot(X_gt(:,1),X_gt(:,2),'r-','LineWidth',5);
hold on
plot(x_gt(:,1),x_gt(:,2),'g+');

%assume frame size of uav 240*240
width = 240;
height = 240;
itrvl = 5;
uav_sim = [];

%% manually selected trajectory key points from aerial iamges
X_trj = [1226,611;
        1144, 764;
        1040, 928;
        914, 1073;
        812, 1173;
        735, 1322;
        464, 1192;
        456, 1041;
        595, 914;
        631, 770;
        821, 673;
        1226,611];
    
%% get trajectory points between key points
X_trj_all = [];
for i = 1:length(X_trj)-1
    X_trj_all = [X_trj_all; X_trj(i,:)];
    x_s = [ X_trj(i,1),X_trj(i,2),1];
    x_e = [ X_trj(i+1,1),X_trj(i+1,2),1];
    l = cross(x_s,x_e);
    if abs(X_trj(i,1)-X_trj(i+1,1))<abs(X_trj(i,2)-X_trj(i+1,2))
        if X_trj(i,2)-X_trj(i+1,2)<0
            for y = X_trj(i,2):itrvl:X_trj(i+1,2)
                x = (-l(2)*y-l(3))/l(1);
                p_tmp = [x,y];
                X_trj_all = [X_trj_all; p_tmp];
            end
        else
            for y = X_trj(i,2):-itrvl:X_trj(i+1,2)
                x = (-l(2)*y-l(3))/l(1);
                p_tmp = [x,y];
                X_trj_all = [X_trj_all; p_tmp];
            end
        end
    else
        if X_trj(i,1)-X_trj(i+1,1)<0
            for x = X_trj(i,1):itrvl:X_trj(i+1,1)
                y = (-l(1)*x-l(3))/l(2);
                p_tmp = [x,y];
                X_trj_all = [X_trj_all; p_tmp];
            end
        else
             for x = X_trj(i,1):-itrvl:X_trj(i+1,1)
                y = (-l(1)*x-l(3))/l(2);
                p_tmp = [x,y];
                X_trj_all = [X_trj_all; p_tmp];
            end
        end
        
    end
end

%% get simulated uav video sequences from aerial images
I1 = imread('ER0354_1500.bmp');
figure(2);
imshow(I1);
hold on
plot(X_trj(:,1),X_trj(:,2),'r-','LineWidth',5);
hold on
% plot(X_trj_all(:,1),X_trj_all(:,2),'b-','LineWidth',2);
% figure(5);
mov(1:length(X_trj_all)) = struct('cdata',[],'colormap',[]);
for i = 1:length(X_trj_all)
    frm = GetOneFrm(X_trj_all(i,1),X_trj_all(i,2),width,height,I1);
    %uav_sim(:,:,i) = frm;
    mov(i) = im2frame(frm);
    %imshow(frm);
    %show trajectory in original image
    plot(X_trj_all(i,1),X_trj_all(i,2),'g+');
    hold on 
    pause(0.01);
end
% figure(3)
% movie(mov,1,30);
% movie2avi(mov,'uav_sim.avi','compression','none');










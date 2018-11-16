figure;
speed_sm=imread('speed_sm.bmp');
speed_am=imread('speed_am.bmp');
speed_ssm=imread('speed_ssm.bmp');

subplot(1, 3, 1), imagesc(speed_sm);
subplot(1, 3, 2), imagesc(speed_am);
subplot(1, 3, 3), imagesc(speed_ssm);

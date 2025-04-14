% 读取图像文件
imagePath = 'IMG_0419-0012.png'; % 替换为你的图片路径
img = imread(imagePath);
img=im2gray(img)
% 显示图像
imshow(img);
title('使用 imread 读取的图像');

rect = [616 412 37 37];

B = imcrop(img, rect);
figure;
imshow(B)
c=['real_dot_of',imagePath(1:13),'.bmp']
imwrite(B,c)


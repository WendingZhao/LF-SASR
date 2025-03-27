% 设置图像尺寸（例如500x500）
size = 256;

% 创建纯黑图像（灰度，uint8类型）
img = zeros(size, size, 'uint8');

% 计算中心点坐标（四舍五入到最近的整数）
center = round((size + 1) / 2);

% 将中心点设置为白色（255）
img(center, center) = 255;

% 显示图像
imshow(img);
imwrite(img,'./dot.bmp')
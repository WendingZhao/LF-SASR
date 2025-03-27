function [] = main(path,path_save)


files = dir(path);
files(1:2) = [];

for index=1:length(files)
    file_name = strsplit(files(index).name, '.');
    file_name = file_name{1};
    load([path, file_name,'.LFR'], 'LF');

    [~, ~, H, W, ~] = size(LF);
    H = fix(H/4) * 4;
    W = fix(W/4) * 4;
    LF = LF(6:10, 6:10, 1:H, 1:W, 1:3);
    [U, V, H, W, ~] = size(LF);
    
    mkdir([path_save, file_name]);
    for i=1:U
        for j=1:V
            tmp = im2uint8(squeeze(LF(i,j,:,:,:)));
            imshow(tmp);
            imwrite(tmp, [path_save, file_name, '/', file_name, '_', num2str(i-1), '_', num2str(j-1), '.bmp']);
        end
    end

end

end
data_dir = '/Users/yikai/Desktop/imagenet-dataset';
fileslist = dir([data_dir filesep '*.bin']);

for i=1:length(fileslist)
    baseCurFileName = fileslist(i).name; % '10000.bin'
    fullCurFileName = fullfile(data_dir, baseCurFileName); % 'full path'
    fp = fopen(fullCurFileName);
    cur_nsample = fread(fp, 1, 'int');
    cur_ndim = fread(fp, 1, 'int');
    X = fread(fp, cur_nsample * cur_ndim, 'double');
    fclose(fp);
    
    cur_feat = reshape(X, [cur_ndim, cur_nsample]);
    cur_one_label = str2double(baseCurFileName(1:end-4));
    cur_label = kron(cur_one_label, ones(cur_nsample, 1));
    save(strcat(baseCurFileName(1:end-4), '_feature.mat'), 'cur_feat');
    save(strcat(baseCurFileName(1:end-4), '_label.mat'), 'cur_label');
    isfile('total_feat.mat')
    
    if i==1
        save('total_feat.mat', 'cur_feat', '-append')
        save('total_label.mat', 'cur_label', '-append')
    else
        save('total_feat.mat', 'cur_feat')
        save('total_label.mat', 'cur_label')
    end
end


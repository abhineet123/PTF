function formatGT(data_name)
in_fname=sprintf('%s.txt', data_name);
out_fname=sprintf('GT/%s.txt', data_name);
k=importdata(in_fname);
dlmwrite(out_fname,k.data, 'delimiter', '\t', 'precision', '%5.2f');

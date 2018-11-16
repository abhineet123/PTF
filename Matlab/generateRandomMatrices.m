mat_sizes=100:100:5000;
mat_count=length(mat_sizes);
mat_cell1=cell(mat_count, 1);
mat_cell2=cell(mat_count, 1);
for i=1:mat_count
    mat_size=mat_sizes(i);
    fprintf('mat_size: %d\n', mat_size);
    mat1=randn(mat_size, mat_size);
    mat2=randn(mat_size, mat_size);
    fname1=sprintf('../C++/matrices/mat%d_1.txt', mat_size);
    fname2=sprintf('../C++/matrices/mat%d_2.txt', mat_size);
    
    dlmwrite(fname1, mat1, 'delimiter', '\t', 'precision','%15.12f');
    dlmwrite(fname2, mat2, 'delimiter', '\t', 'precision','%15.12f');
end
    
    

function[] = saveMatrixB(name, matrix)

fprintf('Saving file %s\n', name);
id = fopen(name, 'w');
fwrite(id, matrix, 'double');
fclose(id);


end
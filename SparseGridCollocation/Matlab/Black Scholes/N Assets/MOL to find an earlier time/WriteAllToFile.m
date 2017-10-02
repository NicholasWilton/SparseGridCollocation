function[] = WriteAllToFile(name, A)

id = fopen(name, 'w');
fwrite(id, A, 'double');
fclose(id);


end
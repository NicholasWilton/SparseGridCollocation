function WriteAllTestToFile( filename, Matrix )

fileID = fopen(filename,'w');
fprintf(fileID,'%f\n',Matrix);
fclose(fileID);

end


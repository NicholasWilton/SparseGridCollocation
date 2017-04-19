
delete(gcp('nocreate'))
parpool local

cMuSIK;

cSIK;

%parpool close
delete(gcp('nocreate'))

Nodes=ones(1,9);

for i=3:11
    Nodes(i-2)=nodesnumber(i,2);
end

Slope_RMS=( log(RMS(2:end)) - log(RMS(1:end-1)) ) ./ ( log(Nodes(2:end)) - log(Nodes(1:end-1)) );
Slope_RMS_s=( log(RMS_s(2:end)) - log(RMS_s(1:end-1)) ) ./ ( log(Nodes(2:end)) - log(Nodes(1:end-1)) );
Slope_MAX=( log(MAX(2:end)) - log(MAX(1:end-1)) ) ./ ( log(Nodes(2:end)) - log(Nodes(1:end-1)) );
Slope_MAX_s=( log(MAX_s(2:end)) - log(MAX_s(1:end-1)) ) ./ ( log(Nodes(2:end)) - log(Nodes(1:end-1)) );

figure;loglog( Nodes , RMS , '-r*', Nodes, RMS_s,'-b*');
xlabel('Nodes');ylabel('RMS');title('MuSIK-C vs SIK-C');
legend('MuSIK-C','SIK-C')

for i=1:length(Nodes)-1
text(Nodes(i+1),RMS(i+1),num2str(Slope_RMS(i)),'FontSize',8)
text(Nodes(i),RMS_s(i),num2str(Slope_RMS_s(i)),'FontSize',8)
end

figure;loglog( Nodes , MAX , '-r*', Nodes, MAX_s,'-b*');
xlabel('Nodes');ylabel('max error');title('MuSIK-C vs SIK-C');
legend('MuSIK-C','SIK-C')

for i=1:length(Nodes)-1
text(Nodes(i+1),MAX(i+1),num2str(Slope_MAX(i)),'FontSize',8)
text(Nodes(i),MAX_s(i),num2str(Slope_MAX_s(i)),'FontSize',8)
end
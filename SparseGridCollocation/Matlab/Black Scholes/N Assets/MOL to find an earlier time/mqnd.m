function [ D, D_first, D_second_half, D_second_diag ] = mqNd( TP, CN, A, C, d_Asset )
% For B-S
% TP for test points, CN for center nodes. 
% First column represents time dimension, Subsequent columns represent spatial dimensions
% A is scale parameter
% C is shape parameter
% d_Asset is dimension of assets

[Num,~]=size(CN);
[N,dimensions]=size(TP);
D = deal(ones(N,Num));
D_first = cell(1,d_Asset);
length_second_order_half = (d_Asset^2 - d_Asset)/2 ; 
D_second_half = cell(1,length_second_order_half );
D_second_diag = cell(1,d_Asset);

%  multiquadric RBF..............
 for j=1:Num
    for d=1:dimensions
        PHI =  sqrt( (C(j)/A(j,d))^2 + (TP(:,d)-CN(j,d)).^2 );

        D(:,j)=D(:,j).*PHI;    
    end
     %PHIt =  sqrt( (C(j,1)/A(j,1))^2 + (TP(:,1)-CN(j,1)).^2 );
     if nargout > 1
     
        %Dt(:,j) = (TP(:,1)-CN(j,1)).*D(:,j)./PHIt ./ PHIt;
        
            for d = 1 : dimensions
                PHIx =  sqrt( (C(j)/A(j,d))^2 + (TP(:,d)-CN(j,d)).^2 );
                D_first{d}(:,j) = TP(:,d) .* ( (TP(:,d)-CN(j,d)).* (D(:,j)./PHIx ./ PHIx) );
                D_second_diag{d}(:,j) = TP(:,d).^2 .* ( 1./PHIx - (TP(:,d)-CN(j,d)).^2./PHIx.^3 ) .* ...
                    D(:,j) ./ PHIx;
            end
            
            if dimensions >= 2
                kk = 0;
                for d = 1 : dimensions
                    PHIx =  sqrt( (C(j)/A(j,d))^2 + (TP(:,d)-CN(j,d)).^2 );
                    for i = d+1 : dimensions
                        kk = kk + 1;
                        PHIi =  sqrt( (C(j)/A(j,i))^2 + (TP(:,i)-CN(j,i)).^2 );
                        D_second_half{kk}(:,j) = TP(:,d).*TP(:,i) .* (TP(:,d)-CN(j,d)).*(TP(:,i)-CN(j,i)) .* ...
                            D(:,j) ./ PHIx ./ PHIx ./ PHIi ./ PHIi;
                    end
                end
            end
            
    end
    
 end
end
function mX = CovsToVecs(Covs, mMean)
%     for i = 1:288
%         Covs(:,:,i) = abs(mapminmax(Covs(:,:,i), 0, 1));
%     end
    

    if nargin == 2
        mRiemannianMean = mMean;
    else
        mRiemannianMean = RiemannianMean(Covs);
    end
    
    mCSR            = mRiemannianMean^(-1/2);
    nCSR            = mRiemannianMean^(1/2);
    
    K  = size(Covs, 3);
    D  = size(Covs, 1);%22
    D2 = D * (D + 1) / 2;
    mX = zeros(D2, K);
    
    mW = sqrt(2) * ones(D) - (sqrt(2) - 1) * eye(D);
    for kk = 1 : K
        Skk      = logm(mCSR * Covs(:,:,kk) * mCSR) .* mW;
        mX(:,kk) = Skk(triu(true(size(Skk))));
    end
    
%     mToep       = toeplitz(1:D);
%     vTeop       = mToep(triu(true(size(mToep))));
%     [~, vOrder] = sort(vTeop);
% %     vOrder      = vOrder(1:43);
%     
%     mX = mX(vOrder,:);
   
end
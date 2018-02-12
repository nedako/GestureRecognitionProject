function newD = Gest_applyPCA(data  , catagory , dim , viz)


stdr = nanstd(data);
M=nanmean(data);
Sa = size(data);
sr = (data-repmat(M,Sa(1),1))./repmat(stdr,Sa(1),1);
[coefs,scores,variances,t2] = pca(sr);


C=cov(sr);
[~,E] = eig(C);


newD=sr*E(:,1:dim);

if viz
    figure('color' , 'white')
    percent_explained = 100*variances/sum(variances);
    xlabel('Principal Component')
    ylabel('Variance Explained (%)')
    pareto(percent_explained)
    figure('color' , 'white')
    gscatter(newD(:,1) , newD(:,2) , catagory)
    xlabel('Principal Component 1')
    ylabel('Principal Component 2')
end
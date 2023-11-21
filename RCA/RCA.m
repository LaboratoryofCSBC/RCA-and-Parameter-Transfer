function [CovsDT,D] =  RCA(Covs)

M{1} = RiemannianMean(cat(3, Covs{:,1}));
M{2} = RiemannianMean(cat(3, Covs{:,2}));
M{3} = RiemannianMean(cat(3, Covs{:,3}));
M{4} = RiemannianMean(cat(3, Covs{:,4}));
M{5} = RiemannianMean(cat(3, Covs{:,5}));
D    = RiemannianMean(cat(3, M{1}, M{2}, M{3}, M{4}, M{5}));

m{1} = LogMap(D, M{1});
m{2} = LogMap(D, M{2});
m{3} = LogMap(D, M{3});
m{4} = LogMap(D, M{4});
m{5} = LogMap(D, M{5});
Ds = LogMap(D, D);

d{1} = D-m{1};
d{2} = D-m{2};
d{3} = D-m{3};
d{4} = D-m{4};
d{5} = D-m{5};


CovsDT = Covs;
for ss = [1 2 3 4 5]
    for ii = 1 : size(Covs, 1)
        S{ii,ss} = LogMap(D, Covs{ii,ss});
        S{ii,ss} = S{ii,ss}+d{ss};     
        CovsDT{ii,ss} = ExpMap(D, S{ii,ss});
    end
end

end

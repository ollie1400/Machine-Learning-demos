% make test data
x = rand(2,500);
y = (x(1,:) + 2.*x(2,:)) > 1;
y = (x(2,:) > 0.3);
y1 = (x(1,:)-0.4).^2 + (x(2,:)-0.4).^2 < 0.15;
y2 = (x(1,:)-0.75).^2 + (x(2,:)-0.75).^2 < 0.02;
y = y1 | y2;

%%
lambda = 0;

% linear by default
g = @(theta,x) theta'*x;
dg = @(theta,x) x;
xp = x;


% make new features to allow more complex decision boundaries
mfi = 7;
switch mfi
    case 1
        % no new features, just x(1,:) and x(2,;)
        mf = @(x,y) [x; y];
    case 2
        % include square terms
        mf = @(x,y) [x; y; x.*x; y.*y];
    case 3
        % include cubic terms
        mf = @(x,y) [x; y; x.*x.*x; y.*y.*y];
    case 4
        % include cubic terms and square terms
        mf = @(x,y) [x; y; x.*x; y.*y; x.*x.*x; y.*y.*y];
    case 5
        % include cross terms
        mf = @(x,y) [x; y; x.*y;];
    case 6
        % include cubic terms, square terms and cross terms
        mf = @(x,y) [x; y; x.*x; y.*y; x.*x.*x; y.*y.*y; x.*y];
    case 7
        % include cubic terms, square terms and cross terms
        mf = @(x,y) [x; y; x.*x; y.*y; x.*x.*x; y.*y.*y; x.*y; x.*x.*y; x.*y.*y];

end
xp = mf(x(1,:), x(2,:));

%%
[theta,finalCost,classifications] = ...
    logisticregression(xp,y,lambda,g,dg);
ezplot(@(x,y) theta'*[1; mf(x,y)],[0 1]);

num1 = sum(y == 1);
num0 = sum(y == 0);
correct1 = sum(classifications == 1 & y == 1);
correct0 = sum(classifications == 0 & y == 0);
fprintf('Final cost = %.3e\n',finalCost);
fprintf('Correct 0 classifications = %d/%d (%.2f)\n', ...
    correct0, ...
    num0, ...
    correct0 / num0);
fprintf('Correct 1 classifications = %d/%d (%.2f)\n', ...
    correct1, ...
    num1, ...
    correct1 / num1);
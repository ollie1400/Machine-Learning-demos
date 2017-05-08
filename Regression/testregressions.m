
x = rand(2,50);
y = (x(1,:) + 2.*x(2,:)) > 1;
y = (x(2,:) > 0.3);
y1 = (x(1,:)-0.5).^2 + (x(2,:)-0.5).^2 < 0.1;
y2 = (x(1,:)-0.7).^2 + (x(2,:)-0.7).^2 < 0.1;
y = y1 | y2;

%%
lambda = 0;

% linear by default
g = @(theta,x) theta'*x;
dg = @(theta,x) x;
xp = x;

% make new features to allow more complex decision boundaries
xp = [x; x(1,:).^2; x(2,:).^2];

%%
[theta,finalCost,classifications] = ...
    logisticregression(xp,y,lambda,g,dg);
ezplot(@(x,y) theta'*[1;x;y;x*x;y*y],[0 1]);

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
function linearregression

% data
x = 1:100;
m = length(x);
y = 3 * x + rand(1, length(x));
y = 5*x.*x + 2*x + 13 + rand(1, length(x));
assignin('base','x',x);
assignin('base','y',y);

% model
theta = rand(3,1);  % parameter vector, start random
h = @(theta,x) theta(1) + theta(2) * x + theta(3) * x .* x;  % the model to fit
dh = @(theta,x) [ones(1,m); x; x.*x];  % differental of the model. Element dh(i,j) = dh(theta,x_j)/dtheta_i

% do gradient descent
alpha = 1e-12;  % learning rate
n = 10000;
lastJ = NaN;
threshold = 0.00001;
converged = 0;
Js = zeros(n,1);
display('Starting gradient descent:');
for i=1:n
    [J,dJ] = costFunction(theta, h, dh, x, y);
    Js(i) = J;
    
    % update
    theta = theta - alpha * dJ;
    
    if ~isnan(lastJ)
        pdiff = (J - lastJ) / J;
        display(sprintf('(%i) Change by %+.2f%%', i, pdiff*100));
        if abs(pdiff) < threshold
            converged = 1;
            break;
        end
    end
    
    lastJ = J;
end

figure;
semilogy(1:n,Js);
xlabel('Iteration of gradient descent');
ylabel('Cost function');
grid on;
display(theta);

switch converged
    case 0
        display('Didn''t converge after 1000 runs');
    case 1
        display('Converged!');
end

figure;
hold on;
plot(x,y,'o');
plot(x, h(theta,x));
legend('Data', 'Fit');

end

function [J, dJ] = costFunction(theta, h, dh, x, y)

% J = (1/2m) * sum((h(x_i) - y_i)^2, i)
% dJ/dt_j = (1/2m) * sum( 2*(dh(x_i,i)/dt_j) * (h(x_i) - y_i), i)
m = length(y);
mt = length(theta);
diffs = (h(theta, x) - y);
J = sum(diffs.^2) / (2*m);
dJ = (1/2*m) * sum( 2.*dh(theta,x).*repmat(diffs, [mt 1]), 2);
end
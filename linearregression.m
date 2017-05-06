function linearregression

% data
x = 1:10;
m = length(x);
y = 3 * x + rand(1, length(x));

% model
theta = rand(2,1);  % parameter vector, start random
h = @(theta,x) theta(1) + theta(2) * x;  % the model to fit
dh = @(theta,x) [ones(1,m); x];  % differental of the model. Element dh(i,j) = dh(theta,x_j)/dtheta_i

% do gradient descent
alpha = 0.0001;  % learning rate
n = 1000;
lastJ = NaN;
threshold = 0.01;
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
        display(sprintf('Change by %+.2f%%', pdiff*100));
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

switch converged
    case 0
        display('Didn''t converge after 1000 runs');
    case 1
        display('Converged!');
end

if converged
    figure;
    hold on;
    plot(x,y,'o');
    plot(x, h(theta,x));
    legend('Data', 'Fit');
end

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
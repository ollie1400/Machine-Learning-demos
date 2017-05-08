function [theta,finalCost,classifications] = logisticregression(x,y,lambda,g,dg)

if nargin < 1
    lambda = 0;
    x = rand(2,50);
    y = (x(1,:) + 2.*x(2,:)) > 1;
    y = (x(2,:) > 0.3);
end
if nargin < 5
    % linear by default
    g = @(theta,x) theta'*x;
    dg = @(theta,x) x;
end

% data
m = size(x,2);
mt = size(x,1);

% plot data
figure;
scatter(x(1,:),x(2,:),36,[y' ~y' zeros(m,1)]);
title('Input data (green = 0, red = 1)');

assignin('base','x',x);
assignin('base','y',y);

% model
theta = rand(mt+1,1);  % parameter vector, start random
%theta = [1; 1; 2;];
h = @(theta,x) 1 ./ (1 + exp(-theta'*x));  % the model to fit

% add the bias unit for each data point
x_w0 = [ones(1,m); x];

% minimisation method:
%  1 - gradient descent
%  2 - fminunc
method = 2;  

switch method
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % gradient descent
    case 1
        % do gradient descent
        alpha = 1;  % learning rate
        n = 100000;
        lastJ = NaN;
        threshold = 0.00001;
        converged = 0;
        Js = zeros(n,1);
        display('Starting gradient descent:');
        drawskip = 1;
        if drawskip > 0
            figure(10220);
            scatter(x(1,:),x(2,:),36,[y' ~y' zeros(m,1)]);
            hold on;
            ph = [];
            %xlim([0 1]);
            %ylim([0 1]);
        end
        for i=1:n
            [J,dJ] = costFunction(theta, lambda, g, dg, x_w0, y);
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

            % plot
            if drawskip > 0 && mod(i,drawskip) == 0
                figure(10220);
                d = 0.5;
                db = (log(d / (1-d)) - theta(2)*x(1,:) - theta(1)) / theta(3);
                if ~isempty(ph)
                    delete(ph);
                end
                ph = plot(x(1,:),db);
                drawnow limitrate;
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
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % fminunc
    case 2
        
        options = optimoptions(@fminunc,'GradObj','on');
        [theta,finalCost] = fminunc(@(t) costFunction(t,lambda,g,dg,x_w0,y), theta, options);
end

figure;
hold on;
classifications = h(theta,x_w0) > 0.5;
d = 0.5;
db = (log(d / (1-d)) - theta(2)*x(1,:) - theta(1)) / theta(3);
scatter(x(1,:),x(2,:),36,[y' ~y' zeros(m,1)]);
scatter(x(1,:),x(2,:),10,[classifications' ~classifications' zeros(m,1)]);
%plot(x(1,:),db);
%legend('Data', 'Fit');
xlim([0 1]);
ylim([0 1]);

end

% calculate cost function
% [theta] parameter vector
% [lambda] regularisation parameter
% [x] the training examples
% [y] the labels
function [J, dJ] = costFunction(theta, lambda, g, dg, x, y)

m = length(y);
h = 1 ./ (1 + exp(-g(theta,x)));

if lambda == 0
    regularisationTerm = 0;
else
    regularisationTerm = (lambda / (2*m)) * sum(theta .* theta);
end

J = (-1/m) * (y * log(h)' + (1 - y) * log(1 - h)') + regularisationTerm;
dJ = (-1/m) * dg(theta,x) * ( y - h)';
end
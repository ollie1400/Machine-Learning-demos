function logisticregression(x,y)

% data
%x = rand(2,50);
m = size(x,2);
mt = size(x,1);

% generate labels
%y = (x(1,:).^2 + x(2,:).^2) > 0.5;
%y = (x(1,:) + 2.*x(2,:)) > 1;

% plot data
figure;
scatter(x(1,:),x(2,:),36,[y' ~y' zeros(m,1)]);

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
method = 1;  

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
        drawskip = 0;
        if drawskip > 0
            figure(10220);
            scatter(x(1,:),x(2,:),36,[y' ~y' zeros(m,1)]);
            hold on;
            ph = [];
            %xlim([0 1]);
            %ylim([0 1]);
        end
        for i=1:n
            [J,dJ] = costFunction(theta, x_w0, y);
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
                drawnow;
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
        theta = fminunc(@(t) costFunction(t,x_w0,y), theta, options);
end

figure;
hold on;
py = h(theta,x_w0) > 0.5;
d = 0.5;
db = (log(d / (1-d)) - theta(2)*x(1,:) - theta(1)) / theta(3);
scatter(x(1,:),x(2,:),36,[y' ~y' zeros(m,1)]);
plot(x(1,:),db);
legend('Data', 'Fit');
xlim([0 1]);
ylim([0 1]);

end

function [J, dJ] = costFunction(theta, x, y)

m = length(y);
h = 1 ./ (1 + exp(-theta'*x));
J = (-1/m) * (y * log(h)' + (1 - y) * log(1 - h)');
dJ = (-1/m) * x * ( y - h)';
end
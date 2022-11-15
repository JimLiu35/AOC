figure;

subplot(1,2,1)
plot(out.STATES(:,2), out.STATES(:,3), 'b', 'DisplayName','Trajectory')
hold on
r = 1;
xc = -7;
yc = 0;

theta = linspace(0,2*pi);
x = r*cos(theta) + xc;
y = r*sin(theta) + yc;
plot(x,y,'DisplayName', 'Obstacle')
axis equal
% The first column contains the time points
% The second column contains the state 'x'
legend
title('Trajectory');
xlabel('Px');
ylabel('Py');

subplot(1,2,2)
plot(out.PARAMETERS(:,1) * out.PARAMETERS(1, 2), out.CONTROLS(:,2), 'b', 'DisplayName', 'u1')
hold on;
plot(out.PARAMETERS(:,1) * out.PARAMETERS(1, 2), out.CONTROLS(:,3), 'r', 'DisplayName', 'u2')
title('Control Signal');
legend


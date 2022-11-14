figure;

subplot(1,2,1)
plot(out.STATES(:,2), out.STATES(:,3), 'b')
% The first column contains the time points
% The second column contains the state 'x'
title('x');

subplot(1,2,2)
plot(out.CONTROLS(:,1), out.CONTROLS(:,2), 'b', 'DisplayName', 'u1')
hold on;
plot(out.CONTROLS(:,1), out.CONTROLS(:,3), 'r', 'DisplayName', 'u2')
title('u');
legend


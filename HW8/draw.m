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

figure;

subplot(5,1,1)
plot(out.STATES(:,1), out.STATES(:,2))
xlabel('Time (s)')
ylabel('Px')
subplot(5,1,2)
plot(out.STATES(:,1), out.STATES(:,3))
xlabel('Time (s)')
ylabel('Py')
subplot(5,1,3)
plot(out.STATES(:,1), out.STATES(:,4))
xlabel('Time (s)')
ylabel('$$\theta$$', 'Interpreter','latex')
subplot(5,1,4)
plot(out.STATES(:,1), out.STATES(:,5))
xlabel('Time (s)')
ylabel('v')
subplot(5,1,5)
plot(out.STATES(:,1), out.STATES(:,6))
xlabel('Time (s)')
ylabel('$$\delta$$', 'Interpreter','latex')
sgtitle('States vs Time')

figure;

subplot(2,1,1)
plot(out.CONTROLS(:,1), out.CONTROLS(:,2))
xlabel('Time (s)')
ylabel('$$u_a$$', 'Interpreter','latex')
subplot(2,1,2)
plot(out.CONTROLS(:,1), out.CONTROLS(:,3))
xlabel('Time (s)')
ylabel('$$u_\delta$$', 'Interpreter','latex')
sgtitle('Control Signals vs Time')

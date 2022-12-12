clear all;
close all;
% 2D case
syms x y theta xr yr xin yin alpha l real

Pa = [x; y];
Pr = [xr; yr];
% theta = atan2(yr-y, xr-x);
xin = l * cos(theta) + x;
yin = l * sin(theta) + y;
Pb = [xin; yin];
state = [x; y; theta];
% state = [x; y];
Pba = Pb - Pa;
% Pba_norm = norm(Pba);
Pba_uni = [cos(theta); sin(theta)];
Pra = Pr - Pa;
Pra_norm = norm(Pra);
Pra_uni = Pra / Pra_norm;

Lcon1 = simplify(1 - Pba_uni' * Pra_uni);
% Lcon1x = simplify(diff(Lcon1, state(1)) + diff(Lcon1, state(2)) +...
%     diff(Lcon1, state(3)));
Lcon1x = simplify(jacobian(Lcon1, state'));
% Lcon1xx = simplify(diff(Lcon1x, state(1)) + diff(Lcon1x, state(2)) + ...
%     diff(Lcon1x, state(3)));
Lcon1xx = simplify(jacobian(Lcon1x, state'));
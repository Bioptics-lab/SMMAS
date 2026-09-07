function [x_mean, fval] = Motion3Dsolve(x_motion, y_motion, kx, ky)
%MOTION3DSOLVE  Recover 3D motion from four-view 2D shifts.
%   [X_MEAN, FVAL] = MOTION3DSOLVE(X_MOTION, Y_MOTION, KX, KY)
%   solves for [X; Y; Z] from apparent x/y motion in four views and the
%   corresponding PSF shear coefficients KX and KY.

% Overdetermined residual for all four views (kept; unused by fsolve)
fun = @(x) [x(1) - x_motion(1) + x(3) * kx(1);
    x(2) - y_motion(1) + x(3) * ky(1);
    x(1) - x_motion(2) + x(3) * kx(2);
    x(2) - y_motion(2) + x(3) * ky(2);
    x(1) - x_motion(3) + x(3) * kx(3);
    x(2) - y_motion(3) + x(3) * ky(3);
    x(1) - x_motion(4) + x(3) * kx(4);
    x(2) - y_motion(4) + x(3) * ky(4)];

% Eight square 3-equation subsets used by fsolve
fun1 = @(x) [
    x(1) - x_motion(1) + x(3) * kx(1);
    x(2) - y_motion(1) + x(3) * ky(1);
    x(1) - x_motion(2) + x(3) * kx(2)];
fun2 = @(x) [
    x(2) - y_motion(1) + x(3) * ky(1);
    x(1) - x_motion(2) + x(3) * kx(2);
    x(2) - y_motion(2) + x(3) * ky(2)];
fun3 = @(x) [
    x(1) - x_motion(2) + x(3) * kx(2);
    x(2) - y_motion(2) + x(3) * ky(2);
    x(1) - x_motion(3) + x(3) * kx(3)];
fun4 = @(x) [
    x(2) - y_motion(2) + x(3) * ky(2);
    x(1) - x_motion(3) + x(3) * kx(3);
    x(2) - y_motion(3) + x(3) * ky(3)];
fun5 = @(x) [
    x(1) - x_motion(3) + x(3) * kx(3);
    x(2) - y_motion(3) + x(3) * ky(3);
    x(1) - x_motion(4) + x(3) * kx(4)];
fun6 = @(x) [
    x(2) - y_motion(3) + x(3) * ky(3);
    x(1) - x_motion(4) + x(3) * kx(4);
    x(2) - y_motion(4) + x(3) * ky(4)];
fun7 = @(x) [
    x(1) - x_motion(4) + x(3) * kx(4);
    x(2) - y_motion(4) + x(3) * ky(4);
    x(1) - x_motion(1) + x(3) * kx(1)];
fun8 = @(x) [
    x(2) - y_motion(4) + x(3) * ky(4);
    x(1) - x_motion(1) + x(3) * kx(1);
    x(2) - y_motion(1) + x(3) * ky(1)];

% Solve each subset with fsolve, then average
x = [];
x0 = [10; 10; 10];
options = optimoptions('fsolve', 'Display', 'off');
[x(:, 1), fval, exitflag] = fsolve(fun1, x0, options);
[x(:, 2), fval, exitflag] = fsolve(fun2, x0, options);
[x(:, 3), fval, exitflag] = fsolve(fun3, x0, options);
[x(:, 4), fval, exitflag] = fsolve(fun4, x0, options);
[x(:, 5), fval, exitflag] = fsolve(fun5, x0, options);
[x(:, 6), fval, exitflag] = fsolve(fun6, x0, options);
[x(:, 7), fval, exitflag] = fsolve(fun7, x0, options);
[x(:, 8), fval, exitflag] = fsolve(fun8, x0, options);

% Report the mean solution from the last fsolve exit flag
if exitflag > 0
    disp('Solution of the equation set:')
    disp(mean(x, 2))
else
    disp('Solver failed')
end
x_mean = mean(x, 2);
end

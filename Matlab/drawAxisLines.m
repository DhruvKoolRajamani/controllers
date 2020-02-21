function [p] = drawAxisLines(fig, rot, p_in)
%DRAWAXISLINES Summary of this function goes here
%   Detailed explanation goes here
x = [1 0 0].';
y = [0 1 0].';
z = [0 0 1].';
rX = rot*x;
rY = rot*y;
rZ = rot*z;
rX = 0.1*(rX(:,1)) + p_in;
rY = 0.1*(rY(:,1)) + p_in;
rZ = 0.1*(rZ(:,1)) + p_in;
ax_x = [p_in(:,1), rX(:,1)];
ax_y = [p_in(:,1), rY(:,1)];
ax_z = [p_in(:,1), rZ(:,1)];
hold on
xlabel('X');
ylabel('Y');
zlabel('Z');
plot3(ax_x(1,:), ax_x(2,:), ax_x(3,:), 'r', 'LineWidth',2);
hold on
plot3(ax_y(1,:), ax_y(2,:), ax_y(3,:), 'g', 'LineWidth',0.5);
plot3(ax_z(1,:), ax_z(2,:), ax_z(3,:), 'b', 'LineWidth',2);
end
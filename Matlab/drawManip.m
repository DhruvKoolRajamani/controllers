function [f] = drawManip(Transformations, ax)
%DRAWMANIP Summary of this function goes here
%   Detailed explanation goes here
[l,m,n] = size(Transformations);
trans = 1;
if n > 1
    trans = 1;
else
    trans = 0;
    n = m;
end
f = gcf;
rotate3d on
grid on
xlabel('X');
ylabel('Y');
zlabel('Z');
R1 = [1 0 0;
      0 1 0;
      0 0 1];
if trans == 1
    x = [0, Transformations(1,4,1)];
    y = [0, Transformations(2,4,1)];
    z = [0, Transformations(3,4,1)];
    
else
    x = [0, Transformations(1,1)];
    y = [0, Transformations(2,1)];
    z = [0, Transformations(3,1)];
end
plot3(x, y, z,'m','LineWidth',3,'MarkerEdgeColor', 'b');
hold on
if trans == 1
    drawAxisLines(f,R1,[0;0;0]); % indTransforms(1:3,1:3,1) Transformations(1:3,4,1)
end
view(ax);
axis('equal');
for i = 2:n
    if trans == 1
        x = [Transformations(1,4,i-1), Transformations(1,4,i)];
        y = [Transformations(2,4,i-1), Transformations(2,4,i)];
        z = [Transformations(3,4,i-1), Transformations(3,4,i)];
    else
        x = [Transformations(1,i-1), Transformations(1,i)];
        y = [Transformations(2,i-1), Transformations(2,i)];
        z = [Transformations(3,i-1), Transformations(3,i)];
    end
    if trans == 1
        drawAxisLines(f,Transformations(1:3,1:3,i-1),Transformations(1:3,4,i-1));
    end
    plot3(x, y, z,'m','LineWidth',3, 'MarkerEdgeColor', 'b');
    hold on
end

if trans == 1
    drawAxisLines(f,Transformations(1:3,1:3,n),Transformations(1:3,4,n));
end
hold off
end


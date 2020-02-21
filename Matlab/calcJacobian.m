function [Jv, Jw, p, z] = calcJacobian(Transforms,q,manip,vals)
%CALCJACOBIAN Summary of this function goes here
%   Detailed explanation goes here
solve_flag = true;
[vm, vn] = size(vals);
if vm <= 2
    solve_flag = false;
end

[tm, tn, ti] = size(Transforms);
p = sym(zeros(3, ti + 1));
z = sym(zeros(3, ti + 1));
joint = cellstr(manip')';
[m,n] = size(joint);
ch = joint(m,1);
if lower(ch{1}) == 'r'
    z(1:3,1) = [0 0 1].';
else
    z(1:3,1) = [0 0 0].';
end
for i=1:ti
    p(:,i+1) = Transforms(1:3,4,i);
    z(:,i+1) = Transforms(1:3,3,i);
end
[pm, pn] = size(p);
Jv = sym(zeros(3,ti,n));
Jw = sym(zeros(3,ti,n));

for i=2:pn
    k = i-1;
    char = joint(m,k);
    switch lower(char{1})
        case 'r' % revolute joints
            Jw(:,k,k) = z(:,k);
            if k>1
                for j = 1:k-1
                    Jw(:,j,k) = Jw(:,j,j);
                end
            end
        case 'p' % prismatic joints
            Jw(:,k,k) = 0;
            if i>1
                for j = 1:k-1
                    Jw(:,j,k) = Jw(:,j,j);
                end
            end
    end
end

Jv = sym(zeros(3,ti,ti));
for k=1:ti
    for i=1:3
        for j=1:ti
            if q(j) == 0
                Jv(i,j,k) = 0;
            else
                Jv(i,j,k) = diff(Transforms(i,4,k),q(j));
            end
        end
    end
end

if solve_flag
    Jv = subs(Jv,[q;pi],[vals;3.1416]);
    Jw = subs(Jw,[q;pi],[vals;3.1416]);
end

end


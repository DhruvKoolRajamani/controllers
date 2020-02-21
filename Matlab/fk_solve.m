function [T] = fk_solve(Transforms, states, vals, n)
%FK_SOLVE This function substitutes the states in the transformation matrix
%with the values specified while maintaining the joint constraints imposed
%by the ABB Arm
%   @param transforms is the 4x4xm transformation matrices to solve
%   @param states are the states
%   @param vals are the values required
%   @param n is the T of n wrt 0 with substituted values.
% 

q = vals;

% if abs(q(1,1)) > 165
%     disp('Joint 1 has reached max limit')
%     if q(1,1) > 0
%         q(1,1) = 165;
%     else
%         q(1,1) = -165;
%     end
% end
% 
% if abs(q(2,1)) > 110
%     disp('Joint 2 has reached max limit')
%     if q(2,1) > 0
%         q(2,1) = 110;
%     else
%         q(2,1) = -110;
%     end
% end
% 
% if q(3,1) > 70 || q(3,1) < -110
%     disp('Joint 3 has reached max limit')
%     if q(3,1) > 0
%         q(3,1) = 70;
%     else
%         q(3,1) = -110;
%     end
% end
% 
% if abs(q(4,1)) > 160
%     disp('Joint 4 has reached max limit')
%     if q(4,1) > 0
%         q(4,1) = 160;
%     else
%         q(4,1) = -160;
%     end
% end
% 
% if abs(q(5,1)) > 120
%     disp('Joint 5 has reached max limit')
%     if q(5,1) > 0
%         q(5,1) = 120;
%     else
%         q(5,1) = -120;
%     end
% end
% 
% if abs(q(6,1)) > 400
%     disp('Joint 6 has reached max limit')
%     if q(6,1) > 0
%         q(6,1) = 400;
%     else
%         q(6,1) = -400;
%     end
% end

Transforms = subs(Transforms,[states], [q]);
if n == 0
    T = Transforms(:,:,:);
else
    T = Transforms(:,:,n);
end
end
function [trans] = dhparam2matrix(theta, d, a, alpha)
    %DHPARAM2MATRIX This function returns a Homogenous Transformation matrix
    % given the input parameters a, alpha, theta and d derived from the
    % DH Parameters of the robot.
    %   Please enter all values in degrees and not radians.
    
    cos_alpha = @cos;
    sin_alpha = @sin;
    cos_theta = @cos;
    sin_theta = @sin;
    
    a11 = simplify(cos_theta(theta));
    a12 = simplify(-sin_theta(theta)*cos_alpha(alpha));
    a13 = simplify(sin_theta(theta)*sin_alpha(alpha));
    a14 = simplify(a*cos_theta(theta));
    a21 = simplify(sin_theta(theta));
    a22 = simplify(cos_theta(theta)*cos_alpha(alpha));
    a23 = simplify(-cos_theta(theta)*sin_alpha(alpha));
    a24 = simplify(a*sin_theta(theta));
    a31 = 0;
    a32 = sin_alpha(alpha);
    a33 = cos_alpha(alpha);
    a34 = d;
    scl = 1;
    
    trans = [a11 a12 a13 a14;
             a21 a22 a23 a24;
             a31 a32 a33 a34;
             0   0   0   scl];
end    
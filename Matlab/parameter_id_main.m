clc;
clear all;
close all;


x_MCP = [-1.48353:0.01:0];
x_PIP = [-1.13446:0.01:0];
X_MCP = (x_MCP+1.48353)*(180/pi);
X_PIP = (x_PIP+1.13446)*(180/pi);
V_MCP = X_MCP;
V_PIP = X_PIP;
[mX_MCP, nX_MCP] = size(X_MCP);
[mX_PIP, nX_PIP] = size(X_PIP);

for i=1:nX_MCP
    if i>1
        V_MCP(i) = X_MCP(i) - X_MCP(i-1);
    else
        V_MCP(i) = X_MCP(i);
    end
end

for i=1:nX_PIP
    if i>1
        V_PIP(i) = X_PIP(i) - X_PIP(i-1);
    else
        V_PIP(i) = X_PIP(i);
    end
end

TAU_MCP = 0.0 * X_MCP.^0 + 0.0061639984303806135 * X_MCP.^1 - 0.0005179649292943457 * X_MCP.^2 + 1.3546546119403511e-05 * X_MCP.^3 - 1.5867928748805977e-07 * X_MCP.^4 + 6.903351238054539e-10 * X_MCP.^5 + 0.14548451752943675;
TAU_PIP = 0.0 * X_PIP.^0 + 0.013424611953572195 * X_PIP.^1 - 0.0012670222412968246 *X_PIP.^2 + 4.394356398132201e-05 * X_PIP.^3 - 6.646559645055926e-07 * X_PIP.^4 + 3.668944251673971e-09 * X_PIP.^5 + 0.030560090609476634;

plot(x_MCP,TAU_MCP);
hold on
plot(x_PIP, TAU_PIP);
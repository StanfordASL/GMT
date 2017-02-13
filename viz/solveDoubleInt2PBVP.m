function [path, topt, toff, copt] = solveDoubleInt2PBVP(dim,dt,toff,x0,x1)
% 2d is planar, 3d is with gravity down
% should probably put a max itr counter on the bisection search

topt = topt_bisection(x0,x1,10,1e-4,dim);
T = ceil((topt-toff)/dt);
path = zeros(dim,T);
for ti = 1:T
    t = dt*(ti-1)+toff;
    path(:,ti) = x(t,topt,x0,x1,dim);
end
copt = cost(topt,x0,x1,dim);
toff = (t-topt)+dt;
% if dim == 2
%     plot(path(1,:),path(2,:))
% else
%     plot3(path(1,:),path(2,:),path(3,:))
% end

    function x = x(t,tau,x0,x1,dim)
        if dim == 4
            x = [(x0(1)*(2*t + tau)*(t - tau)^2 + t*(x1(1)*t*(3*tau - 2*t) + ...
                    (t - tau)*tau*(t*(x0(3) + x1(3)) - x0(3)*tau)))/tau^3;
                (x0(2)*(2*t + tau)*(t - tau)^2 + t*(x1(2)*t*(3*tau - 2*t) + ...
                    (t - tau)*tau*(t*(x0(4) + x1(4)) - x0(4)*tau)))/tau^3;
                (6*x0(1)*t*(t - tau) - 6*x1(1)*t*(t - tau) + ...
                    tau*(3*(x0(3) + x1(3))*t^2 - ...
                    2*(2*x0(3) + x1(3))*tau*t + x0(3)*tau^2))/tau^3;
                (6*x0(2)*t*(t - tau) - 6*x1(2)*t*(t - tau) + ...
                    tau*(3*(x0(4) + x1(4))*t^2 - 2*(2*x0(4) + x1(4))*tau*t + ...
                    x0(4)*tau^2))/tau^3];
        elseif dim == 6
            x = [t*x0(4) + x0(1) + (t^3*(2*x0(1) - 2*x1(1) + (x0(4) + x1(4))*tau))/tau^3 - ...
                    (t^2*(3*x0(1) - 3*x1(1) + (2*x0(4) + x1(4))*tau))/tau^2;
                t*x0(5) + x0(2) + (t^3*(2*x0(2) - 2*x1(2) + (x0(5) + x1(5))*tau))/tau^3 - ...
                    (t^2*(3*x0(2) - 3*x1(2) + (2*x0(5) + x1(5))*tau))/tau^2;
                (t*x0(6)*tau^3 + x0(3)*tau^3 + t^2*tau*(-3.*x0(3) + 3.*x1(3) - 2.*x0(6)*tau - 1.*x1(6)*tau) + ...
                     t^3*(2.*x0(3) - 2.*x1(3) + (x0(6) + x1(6))*tau))/tau^3;
                -(2*t*(3*x0(1) - 3*x1(1) + x1(4)*tau))/tau^2 + (3*t^2*(2*x0(1) - 2*x1(1) + x1(4)*tau))/tau^3 + ...
                    (x0(4)*(3*t^2 - 4*t*tau + tau^2))/tau^2;
                -(2*t*(3*x0(2) - 3*x1(2) + x1(5)*tau))/tau^2 + ...
                    (3*t^2*(2*x0(2) - 2*x1(2) + x1(5)*tau))/tau^3 + (x0(5)*(3*t^2 - 4*t*tau + tau^2))/tau^2; 
                (x0(6)*tau^3 + t*tau*(-6.*x0(3) + 6.*x1(3) - 4.*x0(6)*tau - 2.*x1(6)*tau) + ...
                     t^2*(6.*x0(3) - 6.*x1(3) + 3.*x0(6)*tau + 3.*x1(6)*tau))/tau^3];
        end
    end
    
    function topt = topt_bisection(x0,x1,tmax,TOL,dim)
        tu = tmax;
        if dc(tmax,x0,x1,dim) < 0
            topt = tmax;
            return;
        end
        tl = 0.01;
        while dc(tl,x0,x1,dim) > 0
            tl = tl/2;
        end
        topt = 0;
        dcval = 1;
        while abs(dcval) > TOL
            topt = (tu+tl)/2;
            dcval = dc(topt,x0,x1,dim);
            if dcval > 0
                tu = topt;
            else
                tl = topt;
            end
        end
    end

    function dc = dc(tau,x0,x1,dim)
        dtau = 1e-6;
        dc = (cost(tau+dtau/2,x0,x1,dim) - cost(tau-dtau/2,x0,x1,dim))/dtau;
    end

    function c = cost(tau,x0,x1,dim)
        if dim == 4
            c = (1/(tau^3))*(12*x0(1)^2 + 12*x1(1)^2 + 12*x0(2)^2 - 24*x0(2)*x1(2) + ...
                12*x1(2)^2 - 12*x1(1)*(x0(3) + x1(3))*tau + 12*x0(2)*x0(4)*tau - ...
                12*x1(2)*x0(4)*tau + 12*x0(2)*x1(4)*tau - 12*x1(2)*x1(4)*tau + ...
                4*x0(3)^2*tau^2 + 4*x0(3)*x1(3)*tau^2 + 4*x1(3)^2*tau^2 + ...
                4*x0(4)^2*tau^2 + 4*x0(4)*x1(4)*tau^2 + ...
                4*x1(4)^2*tau^2 + tau^4 + 12*x0(1)*(-2*x1(1) + (x0(3) + x1(3))*tau));
        elseif dim == 6
            c = (12*x0(1)^2)/tau^3 + (12*x0(3)^2)/tau^3 + (12*x0(2)^2)/tau^3 - (24*x0(1)*x1(1))/tau^3 + (12*x1(1)^2)/tau^3 - ...
                (24*x0(3)*x1(3))/tau^3 + (12*x1(3)^2)/tau^3 - (24*x0(2)*x1(2))/tau^3 + (12*x1(2)^2)/tau^3 + (12*x0(4)*x0(1))/tau^2 + ...
                (12*x0(5)*x0(2))/tau^2 + (12*x0(2)*x1(5))/tau^2 + (12*x0(1)*x1(4))/tau^2 - (12*x0(4)*x1(1))/tau^2 - (12*x1(4)*x1(1))/tau^2 - ...
                (12*x0(5)*x1(2))/tau^2 - (12*x1(5)*x1(2))/tau^2 + (4*x0(5)^2)/tau + (4*x0(4)^2)/tau + (4*x0(6)^2)/tau + (4*x0(5)*x1(5))/tau + ...
                (4*x1(5)^2)/tau + (4*x0(4)*x1(4))/tau + (4*x1(4)^2)/tau + (4*x1(6)^2)/tau + 101.*tau + ...
                (12.*x0(6)*(x0(3) - 1.*x1(3) + (1/3*x1(6) - 5/3*tau)*tau))/tau^2 + ...
                (12.*x1(6)*(x0(3) - 1.*x1(3) + 5/3*tau^2))/tau^2;
        end
    end
end
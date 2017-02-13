% time optimal dubins car path, 2d (x, y, theta)
function [path, cmin] = dubinsCar(x0,x1,numDisc,v,thetadotBound)
v = x1(1:2) - x0(1:2);
d = norm(v); % distance
th = atan2(v(2),v(1)); % angle between spatial locations
a = mod2piF(x0(3) - th); % angular difference from line
b = mod2piF(x1(3) - th); % angular difference from line

dt = 0.01;

path = [];
cmin = inf;
[path, cmin] = dubinsLSL(d, a, b, cmin, path);
[path, cmin] = dubinsRSR(d, a, b, cmin, path);
[path, cmin] = dubinsRSL(d, a, b, cmin, path);
[path, cmin] = dubinsLSR(d, a, b, cmin, path);
[path, cmin] = dubinsRLR(d, a, b, cmin, path);
[path, cmin] = dubinsLRL(d, a, b, cmin, path);
path = transformPath(path, x0);
end

function path = transformPath(path, xinit)
    th = xinit(3);
    path(:,1:2) = path(:,1:2)*[cos(th) sin(th); -sin(th) cos(th)];
    path = path + repmat(xinit,size(path,1),1);
end
%% Control
function path = carControl(type, d, varargin)
    numDisc = 8;
    if length(varargin) == 1
        numDisc = varargin{1};
    end
    t = linspace(0, d, numDisc)';
    if type < 0 % right turn
        path = [sin(t) cos(t)-1 -t];
    elseif type > 0 % left turn 
        path = [sin(t) -cos(t)+1 t];
    else % straight line
        path = [t zeros(size(t)) zeros(size(t))]; 
    end    
end
%% Word functions
function [path, c] = dubinsLSL(d, a, b, c, path)
ca = cos(a); sa = sin(a); cb = cos(b); sb = sin(b);
tmp = 2 + d*d - 2*(ca*cb + sa*sb - d*(sa - sb));
if tmp < 0
    return;
end
th = atan2(cb - ca, d + sa - sb);
t = mod2piF(-a + th);
p = sqrt(max(tmp, 0));
q = mod2piF(b - th);
cnew = t + p + q;
disp(['LSL: (' num2str(t) ', ' num2str(p) ', ' num2str(q) '), c = ' num2str(cnew)]); 
if (cnew <= c)
    c = cnew;
    path1 = carControl(1, t);
    path2 = transformPath(carControl(0, p), path1(size(path1,1),:));
    path3 = transformPath(carControl(1, q), path2(size(path2,1),:));
    path = [path1; path2; path3];
end
end

function [path, c] = dubinsRSR(d, a, b, c, path)
ca = cos(a); sa = sin(a); cb = cos(b); sb = sin(b);
tmp = 2 + d*d - 2*(ca*cb + sa*sb - d*(sb - sa));
if tmp < 0
    return;
end
th = atan2(ca - cb, d - sa + sb);
t = mod2piF(a - th);
p = sqrt(max(tmp, 0));
q = mod2piF(-b + th);
cnew = t + p + q;
disp(['RSR: (' num2str(t) ', ' num2str(p) ', ' num2str(q) '), c = ' num2str(cnew)]); 
if (cnew <= c)
    c = cnew;
    path1 = carControl(-1, t);
    path2 = transformPath(carControl(0, p), path1(size(path1,1),:));
    path3 = transformPath(carControl(-1, q), path2(size(path2,1),:));
    path = [path1; path2; path3];
end
end

function [path, c] = dubinsRSL(d, a, b, c, path)
ca = cos(a); sa = sin(a); cb = cos(b); sb = sin(b);
tmp = d * d - 2 + 2 * (ca*cb + sa*sb - d * (sa + sb));
if tmp < 0
    return;
end
p = sqrt(max(tmp, 0));
th = atan2(ca + cb, d - sa - sb) - atan2(2, p);
t = mod2piF(a - th);
q = mod2piF(b - th);
cnew = t + p + q;
disp(['RSL: (' num2str(t) ', ' num2str(p) ', ' num2str(q) '), c = ' num2str(cnew)]); 
if (cnew <= c)
    c = cnew;
    path1 = carControl(-1, t);
    path2 = transformPath(carControl(0, p), path1(size(path1,1),:));
    path3 = transformPath(carControl(1, q), path2(size(path2,1),:));
    path = [path1; path2; path3];
end
end

function [path, c] = dubinsLSR(d, a, b, c, path)
ca = cos(a); sa = sin(a); cb = cos(b); sb = sin(b);
tmp = -2 + d * d + 2 * (ca*cb + sa*sb + d * (sa + sb));
if tmp < 0
    return;
end
p = sqrt(max(tmp, 0));
th = atan2(-ca - cb, d + sa + sb) - atan2(-2, p);
t = mod2piF(-a + th);
q = mod2piF(-b + th);
cnew = t + p + q;
disp(['LSR: (' num2str(t) ', ' num2str(p) ', ' num2str(q) '), c = ' num2str(cnew)]); 
if (cnew <= c)
    c = cnew;
    path1 = carControl(1, t);
    path2 = transformPath(carControl(0, p), path1(size(path1,1),:));
    path3 = transformPath(carControl(-1, q), path2(size(path2,1),:));
    path = [path1; path2; path3];
end
end

function [path, c] = dubinsRLR(d, a, b, c, path)
ca = cos(a); sa = sin(a); cb = cos(b); sb = sin(b);
tmp = (6 - d * d  + 2 * (ca*cb + sa*sb + d * (sa - sb))) / 8;
if abs(tmp) >= 1
    return;
end
p = 2*pi - acos(tmp);
th = atan2(ca - cb, d - sa + sb);
t = mod2piF(a - th + p/2);
q = mod2piF(a - b - t + p);
cnew = t + p + q;
disp(['RLR: (' num2str(t) ', ' num2str(p) ', ' num2str(q) '), c = ' num2str(cnew)]); 
if (cnew <= c)
    c = cnew;
    path1 = carControl(-1, t);
    path2 = transformPath(carControl(1, p), path1(size(path1,1),:));
    path3 = transformPath(carControl(-1, q), path2(size(path2,1),:));
    path = [path1; path2; path3];
end
end

function [path, c] = dubinsLRL(d, a, b, c, path)
ca = cos(a); sa = sin(a); cb = cos(b); sb = sin(b);
tmp = (6 - d * d  + 2 * (ca*cb + sa*sb - d * (sa - sb))) / 8;
if abs(tmp) >= 1
    return;
end
p = 2*pi - acos(tmp);
th = atan2(-ca + cb, d + sa - sb);
t = mod2piF(-a + th + p/2);
q = mod2piF(b - a - t + p);
cnew = t + p + q;
disp(['LRL: (' num2str(t) ', ' num2str(p) ', ' num2str(q) '), c = ' num2str(cnew)]); 
if (cnew <= c)
    c = cnew;
    path1 = carControl(1, t);
    path2 = transformPath(carControl(-1, p), path1(size(path1,1),:));
    path3 = transformPath(carControl(1, q), path2(size(path2,1),:));
    path = [path1; path2; path3];
end
end
%% Helper functions
function ang = mod2piF(ang)
    ang = mod(ang, 2*pi);
end

%%
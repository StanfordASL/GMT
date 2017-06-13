%% time optimal dubins airplane, 4d (x,y,z,theta)
% following http://msl.cs.uiuc.edu/~lavalle/papers/ChiLav07b.pdf
% and http://www.et.byu.edu/~beard/papers/preprints/BeardMcLain__.pdf
% min turning radius is 1, max altitude rate is unbounded, velocity xy is 1
% note unbounded altitude unlike previous works, has no effect on
% algorithmic performance (since the paths are computed offline anyway
% except connecting the initial and goal to the tree) and significantly
% simplier implementation

% our cost is distance traveled
function [path, topt] = dubinsAirplane(xinit,xterm,numDisc)
[path, cmin] = dubinsCar(xinit(1:3),xterm(1:3),numDisc,1,1);
dz = xterm(4) - xinit(4);
topt = sqrt(cmin^2 + dz^2);

dd = path(1:size(path,1)-1,1:2)-path(2:size(path,1),1:2);
dd = sum(sqrt(dd.*dd),2);
dist = cumsum(dd);

zpath = xinit(4) + dist/max(dist)*dz;
path = [path [xinit(4); zpath]]
end
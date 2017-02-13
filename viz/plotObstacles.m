function plotObstacles(obstacles)
for i=1:2:length(obstacles(:,1))
    a = obstacles(i,1);
    b = obstacles(i,2);
    c = obstacles(i+1,1);
    d = obstacles(i+1,2);

    fill([a a c c], [d b b d], 'r','EdgeColor','k','facealpha',.4);
end
end


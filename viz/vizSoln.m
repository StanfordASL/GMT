% Author: brian ichter
% Solution visualization
% Currently setup for a double integrator, but just substitude the local
% planner for dubins car for solveDoubleInt2PBVP (the double integrator for
% the double integrator)

solutionData; % put soln.txt output in solutionData.m
format compact; 
edges = edges + 1;
maxCost = max(costs);
dt = 0.025;

[numSamples,dim] = size(samples);

%% obs
obstacles = generateObstacles(3);
dx = 0.003; dy = 0.003; doLabel = false; % label node numbers

%% plot 2D results
close all; figure; hold on;
plotObstacles(obstacles);
for i = 1:numSamples
    color = [costs(i)/maxCost 0 1-costs(i)/maxCost];
    edgeIdx = edges(i);
    if edgeIdx == 0
        plot(samples(i,1), samples(i,2), 'g.');
    elseif edgeIdx == -1
        plot(samples(i,1), samples(i,2), 'm.');
    end
    if edgeIdx > 0
        [path] = solveDoubleInt2PBVP(dim, dt, 0, samples(edgeIdx,:), samples(i,:)); % double int
%         plot([samples(i,1) samples(edgeIdx,1)],[samples(i,2) samples(edgeIdx,2)],'k-d',...
%             'MarkerFaceColor',color,'MarkerEdgeColor',color); % straight line
        plot(path(1,:), path(2,:), 'k-d',...
            'MarkerFaceColor',color,'MarkerEdgeColor',color);
        if doLabel
            text(samples(edges(i),1)+dx, samples(edges(i),2)+dy, num2str(edges(i)));
            text(samples(i,1)+dx, samples(i,2)+dy, num2str(i));
        end
    end
end

% plot the solution
goalIdx = length(edges);
currentEdge = goalIdx;
while edges(currentEdge) > 0
    nextEdge = edges(currentEdge);
    [path] = solveDoubleInt2PBVP(dim, dt, 0, samples(nextEdge,:), samples(currentEdge,:));
    plot(path(1,:), path(2,:), 'g-d',...
            'LineWidth',5,'MarkerSize',5); % double int
        
%     plot([samples(currentEdge,1),samples(nextEdge,1)],...
%         [samples(currentEdge,2),samples(nextEdge,2)],'g-d',...
%         'LineWidth',5,'MarkerSize',5); % straight line
    currentEdge = nextEdge;
end

%% plot 3D and up simulations
if dim > 2
    figure; hold on;
    plot3dObstacles(obstacles);
    for i = 1:0; % turned off because so intensive
        color = [costs(i)/maxCost 0 1-costs(i)/maxCost];
        edgeIdx = edges(i);
        if edgeIdx == 0
            plot3(samples(i,1), samples(i,2), samples(i,3), 'g.');
        elseif edgeIdx == -1
            plot3(samples(i,1), samples(i,2), samples(i,3), 'm.');
        end
        if edgeIdx > 0
            [path] = solveDoubleInt2PBVP(dim, dt, 0, samples(edgeIdx,:), samples(i,:));
            plot3(path(1,:), path(2,:), path(3,:), 'k-o',...
                'MarkerFaceColor',color,'MarkerEdgeColor',color); % double int
%             plot3([samples(i,1) samples(edgeIdx,1)],[samples(i,2) samples(edgeIdx,2)],[samples(i,3) samples(edgeIdx,3)],'k-o',...
%                 'MarkerFaceColor',color,'MarkerEdgeColor',color); % straight line
        end
    end
    
    % plot the solution
    goalIdx = length(edges);
    currentEdge = goalIdx;
    while edges(currentEdge) > 0
        nextEdge = edges(currentEdge);
        samples(currentEdge,:)
        currentEdge
%         plot3([samples(currentEdge,1),samples(nextEdge,1)],...
%             [samples(currentEdge,2),samples(nextEdge,2)],...
%             [samples(currentEdge,3),samples(nextEdge,3)],'g-d',...
%             'LineWidth',5,'MarkerSize',5); % straight line
        [path] = solveDoubleInt2PBVP(dim, dt, 0, samples(nextEdge,:), samples(currentEdge,:));
        plot3(path(1,:), path(2,:), path(3,:), 'g-d',...
            'LineWidth',5,'MarkerSize',5); % double int
        currentEdge = nextEdge;
    end
end
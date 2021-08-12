clc;
clear;
close all;

%% Problem Definition
problem.CostFunction = @(x) Sphere(x); % Cost function
problem.nVar = 5; % 5 D space, could be anything
%VarSize = [1 nVar]; % Matrix size of decision variables(tricky)or solution vector  (see time stamp 07:30)
problem.VarMin = 3; % Lower Bound of Decision Variables
problem.VarMax = 10; % Upper Bound of Decision variables


%% Parameters of PSO
params.MaxIt = 1000; % Max num of iterations
params.nPop = 500; %swarm or population size

params.w = 1; %Inertia coeff.
params.wdamp = 0.99; % Damping Ratio of Inertia Coefficient. Its purpose is to reduce inertia coeff. while the algorithm iterates. If this 
              % was not here, then the algo prematurely converges to a
              % value that is not good enough
           
params.c1 = 2; % Personal acceleration coeff
params.c2 = 2; % Social or global acceleration coeff

params.ShowIterInfo = true; % Show iteration info flag
%% Calling PSO
out = PSO_pro (problem, params);
BestSol = out.BestSol;
BestCosts = out.BestCosts;
%% Results

figure;
%plot(BestCosts, 'LineWidth',2);
semilogy(BestCosts, 'LineWidth',2);
xlabel('Iteration');
ylabel('Best cost');
grid on;
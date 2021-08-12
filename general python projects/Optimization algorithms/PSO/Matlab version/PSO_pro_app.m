clc;
clear;
close all;

%% Problem Definition
problem.CostFunction = @(x) Sphere(x); % Cost function
problem.nVar = 10; % 5 D space, could be anything
%VarSize = [1 nVar]; % Matrix size of decision variables(tricky)or solution vector  (see time stamp 07:30)
problem.VarMin = -10; % Lower Bound of Decision Variables
problem.VarMax = 10; % Upper Bound of Decision variables


%% Parameters of PSO
% Constriction coefficients
kappa = 1;
phi1 = 2.05;
phi2 = 2.05;
phi = phi1+phi2;
chi = 2*kappa/abs(2-phi-sqrt(phi^2-4*phi));


params.MaxIt = 1000; % Max num of iterations
params.nPop = 500; %swarm or population size
params.w = chi; %Inertia coeff.
params.wdamp = 1; % Damping Ratio of Inertia Coefficient. Its purpose is to reduce inertia coeff. while the algorithm iterates. If this 
              % was not here, then the algo prematurely converges to a
              % value that is not good enough           
params.c1 = chi*phi1; % Personal acceleration coeff
params.c2 = chi*phi2; % Social or global acceleration coeff

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
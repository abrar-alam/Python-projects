clc;
clear;
close all;

%% Problem Definition
CostFunction = @(x) myFunction(x); % Cost function
nVar = 1; % 5 D space, could be anything
VarSize = [1 nVar]; % Matrix size of decision variables(tricky)or solution vector  (see time stamp 07:30)
VarMin = -30; % Lower Bound of Decision Variables
VarMax = 1; % Upper Bound of Decision variables


%% Parameters of PSO
MaxIt = 1000; % Max num of iterations
nPop = 500; %swarm or population size

w = 1; %Inertia coeff.
wdamp = 0.99; % Damping Ratio of Inertia Coefficient. Its purpose is to reduce inertia coeff. while the algorithm iterates. If this 
              % was not here, then the algo prematurely converges to a
              % value that is not good enough
           
c1 = 2; % Personal acceleration coeff
c2 = 2; % Social or global acceleration coeff



%% Initialization
%The particle template
empty_particle.Position = [];
empty_particle.Velocity = [];
empty_particle.Cost = [];
empty_particle.Best.Position = [];% Best field also a structure itself!
empty_particle.Best.Cost = [];
%Create population array
particle = repmat(empty_particle, nPop, 1);

% Initialzie global best
GlobalBest.Cost = -inf;

% Initialize population members
for i=1:nPop
    %Generate random solution
    particle(i).Position = unifrnd(VarMin, VarMax, VarSize); % Basically generates 5 component of a 5 variable vaector, because our nVar was 5
    
    %Initialize veleocity
    particle(i).Velocity = zeros(VarSize); % A bit trciky
    %Evaluation
    particle(i).Cost = CostFunction(particle(i).Position);
    
    %Update the Personal Best
    particle(i).Best.Position = particle(i).Position; %tricky
    particle(i).Best.Cost = particle(i).Cost;%tricky
    
    % Update Global Bets
    if particle(i).Best.Cost > GlobalBest.Cost
        GlobalBest = particle(i).Best;
    end
    
end
% Array to hold best cost value in each iteration
BestCosts = zeros(MaxIt,1);
%% Main loop of PSO

for it = 1:MaxIt
    for i=1:nPop
        %Update velocity
        particle(i).Velocity = w*particle(i).Velocity...
            + c1*rand(VarSize).*(particle(i).Best.Position - particle(i).Position)...
            + c2*rand(VarSize).*(GlobalBest.Position - particle(i).Position);
        
        %Update position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        %Evaluation
        particle(i).Cost = CostFunction(particle(i).Position);
        
        %Update Personal best
        if particle(i).Cost > particle(i).Best.Cost
            particle(i).Best.Position = particle(i).Position;
            particle(i).Best.Cost = particle(i).Cost;
            
            % Update Global Bets
            if particle(i).Best.Cost > GlobalBest.Cost
                GlobalBest = particle(i).Best;
            end
        end
        
    end 
    
    %Store the Best COst value
    BestCosts(it) = GlobalBest.Cost;
    
    %Display iteration information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCosts(it))]);
    
    % Damping Inertia Coefficient
    w = w * wdamp; % This line actually reduces the inertia coeff. as this algo iterates
end 
        
%% Results

figure;
%plot(BestCosts, 'LineWidth',2);
semilogy(BestCosts, 'LineWidth',2);
xlabel('Iteration');
ylabel('Best cost');
grid on;
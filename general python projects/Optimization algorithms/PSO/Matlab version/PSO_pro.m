function out = PSO_pro(problem, params)
    %% Problem Definition
    CostFunction = problem.CostFunction; % Cost function
    nVar = problem.nVar; % 5 D space, could be anything
    VarSize = [1 nVar]; % Matrix size of decision variables(tricky)or solution vector  (see time stamp 07:30)
    VarMin = problem.VarMin; % Lower Bound of Decision Variables
    VarMax = problem.VarMax; % Upper Bound of Decision variables


    %% Parameters of PSO
    MaxIt = params.MaxIt; % Max num of iterations
    nPop = params.nPop; %swarm or population size

    w = params.w; %Inertia coeff.
    wdamp = params.wdamp; % Damping Ratio of Inertia Coefficient. Its purpose is to reduce inertia coeff. while the algorithm iterates. If this 
                  % was not here, then the algo prematurely converges to a
                  % value that is not good enough

    c1 = params.c1; % Personal acceleration coeff
    c2 = params.c2; % Social or global acceleration coeff
    
    ShowIterInfo = params.ShowIterInfo; % The Flag for showing the iteration info.
    MaxVelocity = 0.2*(VarMax-VarMin);
    MinVelocity = -MaxVelocity;
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
    GlobalBest.Cost = inf;

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
        if particle(i).Best.Cost < GlobalBest.Cost
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
            
            % Apply Velocity Limits
            particle(i).Velocity = max(particle(i).Velocity, MinVelocity);
            particle(i).Velocity = min(particle(i).Velocity, MaxVelocity);
            %Update position
            particle(i).Position = particle(i).Position + particle(i).Velocity;
            
            % Apply lower and upper bound limits
            particle(i).Position = max (particle(i).Position, VarMin);
            particle(i).Position = min (particle(i).Position, VarMax);

            %Evaluation
            particle(i).Cost = CostFunction(particle(i).Position);

            %Update Personal best
            if particle(i).Cost < particle(i).Best.Cost
                particle(i).Best.Position = particle(i).Position;
                particle(i).Best.Cost = particle(i).Cost;

                % Update Global Bets
                if particle(i).Best.Cost < GlobalBest.Cost
                    GlobalBest = particle(i).Best;
                end
            end

        end 

        %Store the Best COst value
        BestCosts(it) = GlobalBest.Cost;

        %Display iteration information
        if ShowIterInfo
            
            disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCosts(it))]);
        end
        % Damping Inertia Coefficient
        w = w * wdamp; % This line actually reduces the inertia coeff. as this algo iterates
    end 
    out.pop = particle;
    out.GlobalBest = GlobalBest;
    out.BestCosts = BestCosts;
    out.BestSol = GlobalBest.Position
end

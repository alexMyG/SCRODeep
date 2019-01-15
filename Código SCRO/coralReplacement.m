%% Coral Reef replacement
% Function: Performance a selection by a coral reef replacement
% 
% Input:
%     population: set of chromosomes
%     fitness:    fitness of each individual
%     nPobl:      population size
%     poolPopulation: solutions of a pool
%     poolFitness:    fitness of the pool population
%     Natt:           max attempts to replacement
%     
% Output:
%     newPopulation:  selected population
%     newFitness:     fitness of the new population
function [newPopulation,newFitness1] = coralReplacement(population,fitness,nPobl,poolPopulation,poolFitness,Natt)
    newPopulation = population;
    newFitness1 = fitness;
    
    for i=1:numel(poolPopulation(:,1)),
        nAttempts = 0;
        while nAttempts < Natt,
            randPosition = randi([1 nPobl],1);
            if poolFitness(i) > newFitness1(randPosition),
                newPopulation(randPosition,:) = poolPopulation(i,:);
                newFitness1(randPosition) = poolFitness(i);
                nAttempts = Natt+1;
            else
                nAttempts=nAttempts+1;
            end
        end
    end

end
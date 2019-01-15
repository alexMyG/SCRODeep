%% depredation
% Function: depredate a little percentage of a population
% 
% Input:
%     population: population to be depredated
%     fitness1:   fitness of the population
%     Fd:         percentage of the population to be depredated
%     pDep:       probability of depredation
%     
% Output:
%     newPopulation:  depredated population
%     newFitness1:    updated fitness
function [newPopulation, newFitness1] = depredation(population,fitness1,Fd,pDep)
    newPopulation = population;
    newFitness1 = fitness1;
    [sortedFitness, sortedIndexes] = sort(fitness1,'ascend');
    maxValue = round(Fd*numel(find(fitness1~=-1)));
    indStart = find(sortedFitness~=-1);
    
    for i=indStart(1):indStart(1)+maxValue-1,
        if rand()<pDep,
            newPopulation(sortedIndexes(i),:) = -1;
            newFitness1(sortedIndexes(i)) = -1;
        end
    end
end
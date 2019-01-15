%% Asexual selection
% Function: Performance a selection by an asexual procedure
% 
% Input:
%     population: set of chromosomes
%     fitness:    fitness of each individual
%     nPobl:      population size
%     Fa:         percentage of asexual reproduction (selection)
%     
% Output:
%     newPopulation:  selected population
%     newFitness:     fitness of the new population
function [newPopulation, newFitness1] = selectionAsexual(population,fitness1,Fa)
    [sortedFitness, sortedIndexes] = sort(fitness1,'descend');
    minValue = 1;
    maxValue = round(Fa*numel(find(fitness1~=-1)));
    if maxValue ~= 0,
        indx = randi([minValue maxValue],1);
    else
        indx = 1;
    end
    newPopulation = population(sortedIndexes(indx),:);
    newFitness1 = fitness1(sortedIndexes(indx));
        
end
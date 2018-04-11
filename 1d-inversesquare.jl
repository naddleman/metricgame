"Place n agents randomly across the 2-D flat torus and interact"

mutable struct Agent
  loc:: Float64
  strat::Integer
  lab::Integer
end


function tdis(pt1, pt2) #minimum euclidean distance on circle
  perpdist(a, b) = min(abs(mod(a,1) - mod(b,1)),
                       1 - abs(mod(a,1) - mod(b,1)))
  return perpdist(pt1, pt2)
end

function mkagents(n::Integer, p) # agents have [(location), type, label]
  lis = []
  for i in 1:n               # p in [0,1] is probability of first type
    push!(lis,Agent(rand(),rand([1,2]), i))
  end
  return lis
end


function agentdist(a::Agent, b::Agent)
  tdis(a.loc, b.loc)
end

function agentpower(a::Agent, b::Agent, dropoff) #tentatitive strength
  exp(-dropoff * agentdist(a,b))                 #of interaction
end

function agentpower2(a::Agent, b::Agent, dropoff)
  1 / (dropoff * agentdist(a,b))^2
end

function typedsums(list, label, power)
                  #takes [Agent] and a label to find out the
                  # strength of pull from each type
  workinglist = copy(list)
  splice!(workinglist,label)
  sums = [0.0,0.0]
  for agent in workinglist
    if agent.strat == 1
      sums[1] += agentpower2(list[label],agent,power)
    else
      sums[2] += agentpower2(list[label],agent,power)
    end
  end
  return sums
end

function genweightmatrix(list, power) #matrix of agent-agent interaction strength
  matrix = Array{Float64}((length(list), length(list)))
  for i in 1:length(list)
    for j in 1:length(list)
      if i == j
        matrix[i, j] = 0
      else
        matrix[i, j] = agentpower2(list[i], list[j], power)
      end
    end
  end
  return matrix
end

##function updateagent(list, index) # using logit choice

 #matrix not to recalculate interactions
function payofflist(list, powermatrix, game)
  payofflist = zero(Array{Float64}(length(list)))
  strats = Array{Integer}(length(list))
  for i in 1:length(list)
    strats[i] = list[i].strat
  end
  for i in 1:length(list)
    for j in 1:length(list)
      payofflist[i] += powermatrix[i, j] * game[strats[i], strats[j]]
      # sums payoffs from each interaction 
      # O(n^2)
    end
  end
  return payofflist
end

# instead of randomly selecting a neighbor, agents will look at the payoffs
# in the entire population but weight these by interaction strength when
# revising. So interaction strenght features twice, once in calculating 
# payoffs, and again in looking at neighbos to immitate.
# 
# The transition probability for a player switching x<-y is
# W(x<-y) = (1 + exp(-(Py - Px)/K) ^ (-1) 
# from Haubert/Szabo: Game theory and Physics
# where in this case Py is the "other" type and Px is agent's same type
#
# ...Or not. Start with best response, then add noisy BR

function brupdate(label, list, powermatrix, gamematrix) #Best response update
  util1, util2 = 0, 0
  for i in 1:length(list)
    util1 += gamematrix[1, list[i].strat] * powermatrix[label, i]
    util2 += gamematrix[2, list[i].strat] * powermatrix[label, i]
  end
  if util1 == util2
    list[label].strat = rand([1,2])
  elseif util1 > util2
    list[label].strat = 1
  else
    list[label].strat = 2
  end
  return list
end

function getproportion(agentlist)
  popsize = length(agentlist)
  count = 0
  for i in 1:popsize
    if agentlist[i].strat == 1
      count += 1
    end
  end
  return (count / popsize)
end

function splitpops(agentlist)
  type1s, type2s = [], []
  for i in agentlist
    if i.strat == 1
      push!(type1s, i.loc)
    else
      push!(type2s, i.loc)
    end
  end
  return (type1s, type2s)
end

#arguments: # agents, game, distance discounting, Pr(strat =1), # iterations
function runsimulationbr(numagents, payoffs, distancepower, prob, epochs)
  iterations = epochs * numagents
  population = mkagents(numagents, prob)
  weights = genweightmatrix(population, distancepower)
  #println("starting condition:")
  #println(population)
  print("proportion playing 1:")
  println(getproportion(population))
  # for generating colored plot of agent locations
  typelocations = [splitpops(population)]
  for i in 1:iterations
    if i % numagents == 0
      @printf "epoch %d: " (i/numagents)
      println(getproportion(population))
      push!(typelocations, splitpops(population))
    end
    brupdate(rand(1:numagents), population, weights, payoffs)
  end
  #println(population)
  println("simulation successful!")
  #plot(typelocations[1][1], seriestype=:scatter)
  #plot!(typelocations[1][2], seriestype=:scatter)
  return (typelocations, getproportion(population)) 
end

function multirun(numagents, payoffs, distancepower, prob, epochs, runs)
  proportions = []
  for i in 1:runs
    push!(proportions, runsimulationbr(numagents, payoffs, distancepower,
                                     prob, epochs)[2])
  end
  x = length(proportions)
  y = 0
  heterofraction = 0
  for i in proportions
    println(i)
    if !(i == 0 || i == 1)
      y += 1
    end
  end
  return (proportions, y/length(proportions))
end

function popsizesweep(popsizelist, payoffs, distancepower,
                      prob, epochs, runs)
  heterogeneousfraction = []
  proportionslist = []
  for size in popsizelist
    push!(proportionslist, (size, multirun(size, payoffs, distancepower, prob,
                                           epochs, runs))[2])
  end
  for i in proportionslist
    println(i)
  end
end

# locationarray is an array of pairs of lists of coordinates :(
using Plots
pyplot()

function locationgif(locationarray)
  @gif for i in locationarray
    plot(map(a -> a[1], i[1]),map(a -> a[2], i[1]), seriestype=:scatter,
         leg=false)
    plot!(map(a -> a[1], i[2]),map(a -> a[2], i[2]), seriestype=:scatter,
          m=:x, leg=false)
  end
end

##Constants
power = 1
game = [-1 -1; 1 1]
PD = [2 0; 2.5 0.5]
SH = [4 0; 3 3]
coord = [1 0; 0 1]
popsizes = [4, 7, 10, 13, 16, 19, 22, 25]



#multirun(50, coord, power, 0.5, 30, 100)
#typelocs = runsimulationbr(10, coord, power, 0.5, 35)
#locationgif(typelocs)
#println(typelocs[1][1])
#plot(typelocs[1][1], seriestype=:scatter)
popsizesweep(popsizes, coord, power, 0.5, 20, 30)


#function popsizesweep(popsizelist, payoffs, distancepower,
#                      prob, epochs, runs)
#testagents = mkagents(5, 0.5)
#weightmat = genweightmatrix(testagents, power)
#println(testagents)
#brupdate(1, testagents, weightmat, game)
#println(testagents)
#brupdate(2, testagents, weightmat, game)
#println(testagents)




##println(agentdist(testagents[1],testagents[2]))
#println(typedsums(testagents, 1, power))
#println(length(testagents))
#println(weightmat)
#println(payofflist(testagents, weightmat, game))
##println(typeof(testagents))
##println(typeof(testagents[1].loc))

"Place n agents randomly across N-D flat torus to check if HNE possible"

mutable struct Agent
  loc:: Float64
  strat::Integer
  lab::Integer
end

mutable struct Agent2d
  loc::Tuple{Float64, Float64}
  strat::Integer
  lab::Integer
end

mutable struct Agent3d
  loc::Tuple{Float64, Float64, Float64}
  strat::Integer
  lab::Integer
end

function perpdist(pt1, pt2)
  return min(abs(mod(pt1,1) - mod(pt2,1)), 1 - abs(mod(pt1,1) - mod(pt2,1)))
end

function tdis(pt1, pt2) #minimum euclidean distance on circle
  return perpdist(pt1, pt2)
end

function tdis2(pt1, pt2) #minimum euclidean distance on circle
  return sqrt(perpdist(pt1[1],pt2[1])^2 + perpdist(pt1[2],pt2[2])^2)
end

function tdis3(pt1, pt2) #minimum euclidean distance on circle
  return sqrt(perpdist(pt1[1],pt2[1])^2 + perpdist(pt1[2],pt2[2])^2
              + perpdist(pt1[3],pt2[3])^2)
end

agenttypes = [Agent, Agent2d, Agent3d]

distancefns = [tdis, tdis2, tdis3]

function mkagents(n::Integer, p, dim) # agents have [(location), type, label]
  lis = []
  if dim == 1
    for i in 1:n               # p in [0,1] is probability of first type
      push!(lis,agenttypes[dim](rand(),rand([1,2]), i))
    end
  elseif dim == 2
    for i in 1:n
      push!(lis, agenttypes[dim]((rand(),rand()),rand([1,2]), i))
    end
  elseif dim == 3 
    for i in 1:n
      push!(lis, agenttypes[dim]((rand(),rand(),rand()),rand([1,2]), i))
    end
  end
  return lis
end

function agentdist(a, b, dim)
  distancefns[dim](a.loc, b.loc)
end

## alpha is unused here
function expdecay(a, b, dropoff,dim) #tentatitive strength
  exp(-dropoff * agentdist(a,b, dim))                 #of interaction
end

function inversepower(a, b, dropoff, dim)
  1 / (dropoff * agentdist(a,b,dim))^alpha
end

function altipl(a, b, dropoff, dim)
  1 / (1 + (dropoff * agentdist(a,b,dim))^alpha)
end

function linear(a, b, dropoff, dim)
  dropoff*(1 - 2*agentdist(a,b, dim))
end

# functions have type agent -> agent -> Re+ -> Re+ -> Re+
decayfns = Dict{String, Function}("expdecay" => expdecay,
                                  "IPL" => inversepower,
                                  "alt-IPL" => altipl,
                                  "linear" => linear)

function typedsums(list, label, power, func, dim)
                  #takes [Agent] and a label to find out the
                  # strength of pull from each type
  workinglist = copy(list)
  splice!(workinglist,label)
  sums = [0.0,0.0]
  for agent in workinglist
    if agent.strat == 1
      sums[1] += func(list[label],agent,power, dim)
    else
      sums[2] += func(list[label],agent,power, dim)
    end
  end
  return sums
end

function genweightmatrix(list, power, func, dim) #matrix of agent-agent interaction strength
  matrix = Array{Float64}((length(list), length(list)))
  for i in 1:length(list)
    for j in 1:length(list)
      if i == j
        matrix[i, j] = 0
      else
        matrix[i, j] = func(list[i], list[j], power, dim)
      end
    end
  end
  return matrix
end

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

#arguments: # agents, game, distance discounting, Pr(strat =1),
# iterations, decay function
function runsimulationbr(numagents, payoffs, distancepower, prob,
                         epochs, func, dim)
  iterations = epochs * numagents
  population = mkagents(numagents, prob, dim)
  weights = genweightmatrix(population, distancepower, func, dim)
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

function multirun(numagents, payoffs, distancepower, prob, epochs,
                  runs, func, dim)
  proportions = []
  for i in 1:runs
    print("Pop size: ", numagents)
    println("  Run ", i, " out of ", runs)
    push!(proportions, runsimulationbr(numagents, payoffs, distancepower,
                                     prob, epochs, func, dim)[2])
  end
  x = length(proportions)
  y = 0
  heterofraction = 0
  for i in proportions
    #println(i)
    if !(i == 0 || i == 1)
      y += 1
    end
  end
  return (proportions, y/length(proportions))
end

function popsizesweep(popsizelist, payoffs, distancepower,
                      prob, epochs, runs, func, dim)
  heterogeneousfraction = []
  proportionslist = []
  for size in popsizelist
    push!(proportionslist, (size, multirun(size, payoffs, distancepower, prob,
                                           epochs, runs, func, dim))[2])
  end
  println("\n######\ngamma = ", distancepower)
  println("epochs = ", epochs)
  println("runs per population size = ", runs)
  for i in 1:length(proportionslist)
    push!(heterogeneousfraction, proportionslist[i][2])
    println("population size: ", popsizelist[i], " HNE proportion: ",
            proportionslist[i][2])
  end
  #for i in proportionslist
  #  println(i)
  #end
  return (popsizelist, heterogeneousfraction)
end


function writeresults(popsizelist, payoffs, gammalist,
                      prob, epochs, runs, func, dim)
  csvtitle = string(Dates.format(now(), "yyyy-mm-dd"),"-", decayby,
                    "-parameter-sweep.csv")
  popsrow = deepcopy(popsizelist)
  results = Array{Float64}(length(gammalist) + 1, length(popsizelist) + 1)
  results[1, :] = prepend!(popsrow, 0)
  for i in 1:length(gammalist)
    results[i+1,:] = prepend!(popsizesweep(popsizelist, payoffs, gammalist[i],
                                          prob, epochs, runs, func, dim)[2],
                              gammalist[i])
  end
  writecsv(csvtitle, results)
end
# locationarray is an array of pairs of lists of coordinates :(
#using Plots
#pyplot()

#function locationgif(locationarray)
#  @gif for i in locationarray
#    plot(map(a -> a[1], i[1]),map(a -> a[2], i[1]), seriestype=:scatter,
#         leg=false)
#    plot!(map(a -> a[1], i[2]),map(a -> a[2], i[2]), seriestype=:scatter,
#          m=:x, leg=false)
#  end
#end

##Constants
dim = 3
power = 15 
game = [-1 -1; 1 1]
PD = [2 0; 2.5 0.5]
SH = [4 0; 3 3]
coord = [1 0; 0 1]
popsizes = [10, 20, 80]
gammalist = [0.5, 1, 5, 100]
alpha = 2 # exponent for inverse power laws
epochs = 35
runs = 10
decayby = "expdecay"
# ["expdecay", "IPL", "alt-IPL", "linear"]
decayfunc = decayfns[decayby]

#multirun(50, coord, power, 0.5, 30, 100)
#typelocs = runsimulationbr(10, coord, power, 0.5, 35)
#locationgif(typelocs)
#println(typelocs[1][1])
#plot(typelocs[1][1], seriestype=:scatter)

writeresults(popsizes, coord, gammalist, 0.5, epochs, runs, decayfunc, dim)

#=
(pops, proportions) = popsizesweep(popsizes, coord, power, 0.5,
                                   epochs, runs, decayfunc)

using PyPlot
p = scatter(pops, proportions)
# p = plot(pops, proportions, "b-", linewidth=2)
xlabel("population")
ax = gca()
ax[:set_xscale]("log")
ylabel("miscoordination rate")
titlestring = string(decayby, ", alpha: ", alpha, ", gamma: ", power,
                    ", epochs:", epochs, ", runs:", runs)
title(titlestring)
savefig("xxbig-exp-gamma-20.png")
show()
=#



#popsizesweep(popsizelist, payoffs, distancepower, prob, epochs, runs, func)
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

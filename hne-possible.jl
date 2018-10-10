#=Place n agents randomly across N-D flat torus to check if HNE possible=#
using Dates
using DelimitedFiles

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
  return sqrt(perpdist(pt1[1],pt2[1])^2 +
              perpdist(pt1[2],pt2[2])^2)
end

function tdis3(pt1, pt2) #minimum euclidean distance on circle
  return sqrt(perpdist(pt1[1],pt2[1])^2 +
              perpdist(pt1[2],pt2[2])^2 +
              perpdist(pt1[3],pt2[3])^2)
end

agenttypes = [Agent, Agent2d, Agent3d]
distancefns = [tdis, tdis2, tdis3]

#TODO:  get p working with following function
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

function mkagentslocations(locs, p, dim)
  lis = []
  n = length(locs)
  for i in 1:n 
    push!(lis,agenttypes[dim](locs[i],rand([1,2]), i))
  end
  return lis
end

function agentdist(a, b, dim)
  distancefns[dim](a.loc, b.loc)
end

## alpha is unused here
function expdecay(a, b, dropoff, dim, alpha) #tentatitive strength
  exp(-dropoff * agentdist(a,b, dim))                 #of interaction
end

function inversepower(a, b, dropoff, dim, alpha)
  1 / (dropoff * agentdist(a,b,dim))^alpha
end

function altipl(a, b, dropoff, dim, alpha)
  1 / (1 + (dropoff * agentdist(a,b,dim))^alpha)
end

function linear(a, b, dropoff, dim, alpha)
  dropoff*(1 - 2 * agentdist(a,b, dim))
end

function cutoff(a, b, dropoff, dim, alpha)
  if dropoff*agentdist(a, b, dim) < 1
    x = 1
  else
    x = 0
  end
  return x
end

decayfns = Dict{String, Function}("expdecay" => expdecay,
                                  "IPL" => inversepower,
                                  "alt-IPL" => altipl,
                                  "linear" => linear,
                                  "cutoff" => cutoff)

function typedsums(params, list, label, gamma)
                  #takes [Agent] and a label to find out the
                  # strength of pull from each type
  workinglist = copy(list)
  splice!(workinglist,label)
  sums = [0.0,0.0]
  for agent in workinglist
    if agent.strat == 1
        sums[1] += params["func"](list[label],agent, gamma, params["dim"])
    else
        sums[2] += params["func"](list[label],agent, gamma, params["dim"])
    end
  end
  return sums
end

#matrix of agent-agent interaction strength
function genweightmatrix(params, population, gamma, alpha)
  matrix = Array{Float64}(undef, (length(population), length(population)))
  for i in 1:length(population)
    for j in 1:length(population)
      if i == j
          matrix[i, j] = 0
      else
          matrix[i, j] = params["func"](population[i],
                                        population[j],
                                        gamma,
                                        params["dim"],
                                        alpha)
      end
    end
  end
  return matrix
end

#create matrix so don't have to recalculate interactions
function payofflist(list, powermatrix, game)
  payofflist = zero(Array{Float64}(undef, length(list)))
  strats = Array{Integer}(undef, length(list))
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

#Best response update, if indifferent, keep strategy
function brupdate(label, list, weightmatrix, gamematrix)
  util1, util2 = 0, 0
  for i in 1:length(list)
    util1 += gamematrix[1, list[i].strat] * weightmatrix[label, i]
    util2 += gamematrix[2, list[i].strat] * weightmatrix[label, i]
  end
  if util1 < util2
    list[label].strat = 2
  elseif util1 > util2
    list[label].strat = 1
  else
    list[label].strat = list[label].strat
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
function runsimulationbr(params, numagents, gamma, alpha)
    iterations = params["epochs"] * numagents
    population = mkagents(numagents, params["probability"], params["dim"])
    weights = genweightmatrix(params, population, gamma, alpha)
    print("proportion playing 1:")
    println(getproportion(population))
    # for generating colored plot of agent locations
    typelocations = [splitpops(population)]
    for i in 1:iterations
      if i % numagents == 0
        print("epoch ", (i/numagents), ": ")
        println(getproportion(population))
        push!(typelocations, splitpops(population))
      end
      brupdate(rand(1:numagents), population, weights, payoffs)
    end
    println("simulation successful!")
    #plot(typelocations[1][1], seriestype=:scatter)
    #plot!(typelocations[1][2], seriestype=:scatter)
    return (typelocations, getproportion(population)) 
end

# simulates with fixed agent locations instead of making list of agents
function runlocsbr(params, locations, gamma, alpha)
    iterations = params["epochs"] * length(locations)
    numagents = length(locations)
    population = mkagentslocations(locations,
                                   params["probability"],
                                   params["dim"])
    weights = genweightmatrix(params, population, gamma, alpha)
    print("proportion playing 1:")
    println(getproportion(population))
    typelocations = [splitpops(population)]
    for i in 1:iterations
      if i % numagents == 0
        print("epoch ", (i/numagents), ": ")
        println(getproportion(population))
        push!(typelocations, splitpops(population))
      end
      brupdate(rand(1:numagents), population, weights, params["payoffs"])
    end
    println("simulation successful!")
    return (typelocations, getproportion(population))
end


# this initializes locations and runs a series of simulations with different
#initial strategies to see if the arrangment admits an HNE

function testhne(params, numagents, gamma, alpha)
  locs = []
  heterocount = 0
  i = 0
  for i in 1:numagents
      if params["dim"] == 1
        push!(locs, rand())
      elseif params["dim"] == 2
        push!(locs, (rand(), rand()))
      elseif params["dim"] == 3
        push!(locs, (rand(), rand(), rand()))
    end
  end
  for i = 1:params["limit"]
    println("testing arrangement iteration: ", i)
    heterogeneity = runlocsbr(params, locs, gamma, alpha)[2]
    if !(heterogeneity == 0 || heterogeneity == 1)
      heterocount += 1
      break
    end
  end
  if heterocount >= 1
    println("HNE found")
  else
    println("no HNE found")
  end
  return heterocount
end

#TODO: HERE    
# generates a set of locations and tests which proportion admits HNE

function multihne(params, numagents, gamma, alpha)
  hnecount = 0
  for i = 1:params["runs"]
    print("Pop size: ", numagents)
    println("  Run ", i, " out of ", params["runs"])
    hnecount += testhne(params, numagents, gamma, alpha)
  end
  fraction = hnecount / params["runs"]
  println(fraction, " admit HNE")
  return fraction
end

function multirun(params, numagents, gamma)
  proportions = []
  for i in 1:params["runs"]
    print("Pop size: ", numagents)
    println("  Run ", i, " out of ", params["runs"])
    push!(proportions, runsimulationbr(params, numagents, gamma)[2])
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

function popsizesweep(params, specgamma)
  heterogeneousfraction = []
  proportionslist = []
  for size in params["poplist"] 
    push!(proportionslist, (size, multirun(params, size, specgamma))[2])
  end
  println("\n######\ngamma = ", specgamma)
  println("epochs = ", params["epochs"])
  println("runs per population size = ", params["runs"])
  for i in 1:length(proportionslist)
    push!(heterogeneousfraction, proportionslist[i][2])
    println("population size: ", params["poplist"][i], " HNE proportion: ",
            proportionslist[i][2])
  end
  #for i in proportionslist
  #  println(i)
  #end
  return (popsizelist, heterogeneousfraction)
end

function popsizehnesweep(params, specgamma, specalpha)
  hnepossiblefraction = []
  for size in params["poplist"]
      push!(hnepossiblefraction, multihne(params, size, specgamma, specalpha))
  end
  println("\n######\ngamma = ", specgamma)
  println("alpha = ", specalpha)
  println("epochs = ", params["epochs"])
  println("runs per population size = ", params["runs"])
  return (params["poplist"], hnepossiblefraction)
end

function writeresultshne(params) # cycling over gammas, takes first alpha
  csvtitle = string(Dates.format(now(), "yyyy-mm-dd"),
                    "-",
                    decayby,
                    "-dim-",
                    dim,
                    "-HNE-parameter-sweep.csv")
  popsrow = deepcopy(params["poplist"])
  results = Array{Float64}(undef,
                           length(params["gammas"]) + 1,
                           length(params["poplist"]) + 1)
  results[1, :] = prepend!(popsrow, 0)
  for i in 1:length(params["gammas"])
      results[i+1,:] = prepend!(popsizehnesweep(params,
                                                params["gammas"][i],
                                                params["alphas"][1])[2],
                                params["gammas"][i])
  end
  open(csvtitle, "w") do io
      writedlm(io, results, ',')
  end
end

function writeresults(params)
  csvtitle = string(Dates.format(now(), "yyyy-mm-dd"),
                    "-",
                    params["func"],
                    "-dim-",
                    params["dim"],
                    "-parameter-sweep.csv")
  popsrow = deepcopy(params["poplist"])
  results = Array{Float64}(undef,
                           length(params["gammas"]) + 1,
                           length(params["poplist"]) + 1)
  results[1, :] = prepend!(popsrow, 0)
  for i in 1:length(params["gammas"])
      results[i+1,:] = prepend!(popsizesweep(params, params["gammas"][i])[2],
                                params["gammas"][i])
  end
  open(csvtitle, "w") do io
      writedlm(io, results, ',')
  end
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
dim = 1
power = 15 # decay (gamma)
prob = 0.5 #intial proportion playing each strategy
PD = [2 0; 2.5 0.5]
SH = [4 0; 3 3]
coord = [1 0; 0 1]
game = coord
popsizes = [4,12]
gammalist = [2, 6, 10]
alphalist = [2, 2.5, 3]
alpha = 2 # exponent for inverse power laws
epochs = 50
limit = 25 # how many times to try to find an HNE
runs = 10 # runs per population size
hnelimit = 25 # give up finding hne after this pt
decayby = "expdecay" 
# ["expdecay", "IPL", "alt-IPL", "linear", "cutoff"]
decayfunc = decayfns[decayby]

#function writeresults(popsizelist, payoffs, gammalist,
#                      prob, epochs, runs, func, dim, limit)
parameters = Dict{String, Any}("poplist"     => popsizes,
                               "payoffs"     => game,
                               "gammas"      => gammalist,
                               "alphas"      => alphalist,
                               "probability" => prob,
                               "epochs"      => epochs,
                               "runs"        => runs,
                               "func"        => decayfunc,
                               "dim"         => dim,
                               "limit"       => limit
                              )


#multirun(50, coord, power, 0.5, 30, 100)
#
#typelocs = runsimulationbr(10, coord, power, 0.5, 35)
#locationgif(typelocs)
#println(typelocs[1][1])
#plot(typelocs[1][1], seriestype=:scatter)

writeresultshne(parameters)

#writeresultshne(popsizes, coord,
#                gammalist, 0.5,
#                epochs, runs, decayfunc,
#                dim, limit)

#writeresults(popsizes, coord, gammalist, 0.5, epochs, runs, decayfunc,
#                dim, limit)

#testhne(8, coord, 5, 35, 25, decayfunc, dim)
#multihne(popsizes[2], coord, gammalist[3], epochs, limit, decayfunc,
#                  dim, runs)
#writeresults(popsizes, coord, gammalist, 0.5, epochs, runs, decayfunc, dim)
#
#writeresultshne(popsizelist, payoffs, gammalist, prob, epochs, runs,
#                         func, dim, limit)



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
